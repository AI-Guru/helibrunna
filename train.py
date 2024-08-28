# Helibrunna - A HuggingFace compatible xLSTM trainer.
# Copyright (c) 2024 Dr. Tristan Behrens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import datetime
import os
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from dacite import from_dict
from datasets import load_dataset, load_from_disk
import fire
import hashlib
import json
from omegaconf import OmegaConf
import multiprocessing
import shutil
import sys
import tempfile
import time
from tqdm import tqdm
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from source.utilities import display_logo, human_readable_number, load_configs, validate_config, is_torch_compile_ready, model_from_config

# Import the LinearWarmupCosineAnnealing scheduler from the experiments module.
# Source: https://github.com/NX-AI/xlstm/tree/main
if not os.path.exists("experiments/lr_scheduler.py"):
    import urllib.request
    url = "https://raw.githubusercontent.com/NX-AI/xlstm/main/experiments/lr_scheduler.py"
    os.makedirs("experiments", exist_ok=True)
    urllib.request.urlretrieve(url, "experiments/lr_scheduler.py")
from experiments.lr_scheduler import LinearWarmupCosineAnnealing

# 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def main():
    """
    Main function to run the training process.

    Args:
        preprocess (bool): Whether to only preprocess the dataset and tokenizer.
        config_paths (str): The paths to the configuration files.
    Raises:

    Returns:
        None
    """

    # Parse the arguments.
    arguments = sys.argv[1:]

    # Raise an error if no arguments are provided.
    if len(arguments) == 0:
        print("No arguments provided.")
        sys.exit(1)

    # See if the first argument is "preprocess".
    if arguments[0] == "preprocess":
        preprocess = True
        arguments = arguments[1:]
    else:
        preprocess = False

    # All the remaining arguments are configuration files.
    config_paths = arguments

    # Check if any configuration files are provided.
    if len(config_paths) == 0:
        print("No configuration files provided.")
        sys.exit(1)

    # Run preprocessing or training.
    if preprocess:
        preprocess_only(config_paths)
    else:
        run_training(config_paths)


def run_training(config_paths: list[str]):
    """
    Run the training process based on the provided configuration file.
    Args:
        config_path (str): The path to the configuration file.
    Raises:
        FileNotFoundError: If the configuration file is not found.
    Returns:
        None
    """

    # Load the configuration.
    config = load_configs(config_paths)
    #validate_config(config)

    # Specify the output_dir.
    run_dir = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = os.path.join(config.training.output_dir, run_dir)

    # Initialize the loggers.
    loggers = []
    if "wandb_project" in config.training and config.training.wandb_project is not None and config.training.wandb_project != "":
        loggers.append("wandb")

    # Get gradient accumulation steps.
    gradient_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    #config.training.batch_size = config.training.batch_size * gradient_accumulation_steps
    mixed_precision = config.training.get("mixed_precision", None)

    # Initialize the accelerator.
    accelerator = Accelerator(
        log_with=loggers,
        project_dir=output_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )

    # Display the logo.
    if accelerator.is_local_main_process:
        display_logo()

    # Display mixed precision.
    if accelerator.is_local_main_process:
        if mixed_precision is not None:
            print(f"Mixed precision enabled. Precision: {mixed_precision}")
        else:
            print("Mixed precision disabled.")

    # Create the output directory.
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.print(f"Output directory: {output_dir}")

    # Set log every step to save every step.
    if "log_every_step" not in config.training:
        config.training.log_every_step = 1
    if "save_every_step" not in config.training:
        config.training.save_every_step = -1

    # Preprocess the dataset and tokenizer.
    tokenized_datasets, tokenizer = preprocess(config, accelerator)
    
    # Get the fill token and its id.
    fill_token = config.tokenizer.fill_token
    if fill_token is None:
        raise Exception("Fill token is missing.")
    fill_token_id = tokenizer.convert_tokens_to_ids(fill_token)

    # Get the vocabulary size.
    vocab_size = tokenizer.vocab_size
    config.model.vocab_size = vocab_size
        
    # Create the data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create the model.
    accelerator.print("Creating model...")
    model = model_from_config(config.model)
    model = model.to(device=accelerator.device)
    #model.reset_parameters()

    # Apply precision.
    training_dtype = get_torch_dtype(config.training.weight_precision)
    model = model.to(dtype=training_dtype)
    accelerator.print(f"Training dtype: {training_dtype}")

    # Attempt torch compile.
    if config.training.get("torch_compile", True):
        if not is_torch_compile_ready():
            accelerator.print("WARNING: GPU is not torch compile ready. Training may be slower.")
        model = torch.compile(model)
        print("Model compiled.")

    # Print the model.
    accelerator.print(model)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_human = human_readable_number(num_params)
    accelerator.print(f"Number of parameters: {num_params:_} ({num_params_human})")
    
    # Prepare the DataLoader from the tokenized dataset.
    # Each batch will be padded to the maximum length in the batch.
    accelerator.print("Preparing DataLoader...")
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # Estimate the number of steps.
    num_steps = config.training.num_epochs * len(tokenized_datasets["train"]) // config.training.batch_size
    num_steps = num_steps // accelerator.num_processes
    accelerator.print(f"Estimated number of steps: {num_steps:_}")

    # If the lr_decay_until_steps is set to "auto", set it to the number of steps.
    if config.training.lr_decay_until_steps == "auto":
        config.training.lr_decay_until_steps = num_steps

    # If the lr_warmup_steps is a percentage, convert it to a number of steps.
    if isinstance(config.training.lr_warmup_steps, str):
        percentage = config.training.lr_warmup_steps
        if not percentage.endswith("%"):
            raise ValueError(f"Invalid percentage: {percentage}")
        percentage = float(percentage[:-1]) / 100
        config.training.lr_warmup_steps = int(num_steps * percentage)

    # Prepare the optimizer and learning rate scheduler.
    optimizer_groups = create_weight_decay_optim_groups(model)
    optimizer = torch.optim.AdamW(
        (
            {"weight_decay": config.training.weight_decay, "params": optimizer_groups[0]},
            {"weight_decay": 0.0, "params": optimizer_groups[1]},
        ),
        lr=config.training.lr,
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        config.training.lr_warmup_steps,
        config.training.lr_decay_until_steps,
        config.training.lr,
        config.training.lr_decay_factor * config.training.lr,
    )

    # Prepare model, optimizer, and dataloader for accelerator.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Get some parameters.
    save_every_step = config.training.save_every_step
    log_every_step = config.training.log_every_step
    num_epochs = config.training.num_epochs
    enable_mixed_precision = config.training.enable_mixed_precision
    wandb_project = config.training.get("wandb_project", None)  

    # Get a subset of the config that includes only the model.
    model_config = OmegaConf.select(config, "model")

    # Create the readme.
    create_readme(output_dir, config)

    # Get the model name.
    model_name = config.training.model_name

    # Save the config as yaml and delete it.
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    del config

    # Save the tokenizer.
    tokenizer.save_pretrained(output_dir)

    # Enable trackers.
    if wandb_project is not None:
        accelerator.print(f"Enabling wandb logging for project: {wandb_project}")
        config_dict = OmegaConf.to_container(model_config)
        # Add num_params to the config.
        config_dict["num_params"] = num_params
        config_dict["num_params_human"] = num_params_human
        wandb_run = run_dir + "-" + model_name
        accelerator.init_trackers(
            project_name=wandb_project, 
            config=config_dict,
            init_kwargs={"wandb": {"name": wandb_run}}
        )

    # Training loop.
    step = 0
    running_loss = []
    history = {
        "loss": [],
        "lr": [],
        "epoch": [],
        "step": [],
    }
    average_loss = 0.0
    # Add a green progress bar.
    progress_bar = tqdm(total=num_steps, desc="Training", unit="step", colour="GREEN")

    # Ignore tokens during loss calculation.
    ignore_index = -1
    if tokenizer.pad_token is not None:
        ignore_index = tokenizer.pad_token_id
    accelerator.print(f"Ignore index: {ignore_index}")

    # Do the training.
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            # Assuming batch only contains 'input_ids'
            inputs = batch['input_ids'].to(accelerator.device)

            # Get the labels by shifting the inputs. Remove the first token. Fill the last token.
            labels = torch.roll(inputs, -1, dims=1)
            labels[:, -1] = fill_token_id
            
            # Forward pass.
            # Use gradient accumulation.
            with accelerator.accumulate(model):#, torch.autocast(device_type=accelerator.device.type, dtype=training_dtype, enabled=enable_mixed_precision):

                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=ignore_index,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running_loss.append(loss.item())
                average_loss = sum(running_loss) / len(running_loss)
            
            # Next step.
            step += 1

            # Compute epoch with fraction.
            epoch_fraction = epoch + step / len(train_dataloader)

            # Save every step.
            if step % save_every_step == 0 and step > 0 and save_every_step > 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    save_model(accelerator.unwrap_model(model), model_config, checkpoint_dir)

            # Log every step.
            if step % log_every_step == 0 and step > 0 and log_every_step > 0 and accelerator.is_local_main_process:
                
                # Update the log.
                last_lr = lr_scheduler.get_last_lr()[0]
                history["loss"].append(average_loss)
                history["lr"].append(last_lr)
                history["step"].append(step)
                history["epoch"].append(epoch_fraction)
                running_loss = []

                # Log to wandb.
                if wandb_project is not None:
                    accelerator.log({"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction}, step=step)
                
                # Update the progressbar. Use the step as the total. Also display the loss and lr.
                progress_bar.set_postfix({"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction})
                progress_bar.update(log_every_step)

    # End training.
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Print some information.
    accelerator.print(f"Training completed. Epochs: {epoch}, Steps: {step}")

    # Save the last model.
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}-last")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        save_model(accelerator.unwrap_model(model), model_config, checkpoint_dir)

    # Save the history as JSON.
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)


def create_weight_decay_optim_groups(model):

    # If the model has the methor _create_weight_decay_optim_groups, use it. Likely only for xLSTM.
    if hasattr(model, "_create_weight_decay_optim_groups"):
        return model._create_weight_decay_optim_groups()
    
    # Following the implementation of xLSTM, we split the parameters into two groups: decay and no_decay.
    # The decay group contains all parameters except the ones with a shape of 1.
    # The no_decay group contains all parameters with a shape of 1.
    else:
        decay = set()
        no_decay = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.ndim > 1:
                    decay.add(param)
                elif param.ndim == 1:
                    no_decay.add(param)
                else:
                    raise ValueError(f"Unsupported parameter shape: {param.shape}")
        return tuple(decay), tuple(no_decay)


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Returns the corresponding torch.dtype for the given dtype string.

    Args:
        dtype (str): The dtype string.

    Returns:
        torch.dtype: The corresponding torch.dtype.

    Raises:
        ValueError: If the dtype is unknown.
    """

    if dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    

def save_model(model, model_config, output_dir):
    """
    Save the model and its configuration to the specified output directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        model_config (OmegaConf.DictConfig): The configuration of the model.
        output_dir (str): The directory where the model and configuration will be saved.

    Returns:
        None
    """

    # Make sure the folder exists.
    os.makedirs(output_dir, exist_ok=True)

    # Save the model.
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))

    # Save the model configuration as JSON.
    model_config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(model_config, model_config_path)


def create_readme(output_dir, config):
    """
    Create a README file based on a template and provided configuration.
    Args:
        output_dir (str): The directory where the README file will be saved.
        config (dict): The configuration dictionary containing the necessary information.
    Raises:
        FileNotFoundError: If the template or banner file is not found.
    Returns:
        None
    """

    # Load the template.
    template_path = os.path.join("assets", "readmetemplate.md")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    # Load the template.
    with open(template_path, "r") as f:
        readme_text = f.read()

    # Project name.
    model_name = config.training.model_name

    # Configuration convert the configuration to a yaml string.
    configuration = OmegaConf.to_yaml(config)

    # Base model.
    base_model = "None"
    if "base_model" in config.model:
        base_model = config.model.base_model

    # Tags.
    tags = ["NLP"]
    if "tags" in config.model:
        tags = config.model.tags.split(",")
    tags = "\n".join([f"  - {tag}" for tag in tags])

    # Languages.
    languages = ["en"]
    if "languages" in config.model:
        languages = config.model.languages.split(",")
    languages = "\n".join([f"  - {language}" for language in languages])

    # Datasets.
    datasets = [config.dataset.hugging_face_id]
    datasets = "\n".join([f"  - {dataset}" for dataset in datasets])
    
    # License.
    license = "mit"

    # Format the template.
    readme_text = readme_text.format(
        model_name=model_name,
        configuration=configuration,
        base_model=base_model,
        tags=tags,
        languages=languages,
        datasets=datasets,
        license=license,
    )

    # Save the readme.
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    # Copy the banner.
    banner_path = os.path.join("assets", "trainedwithhelibrunna.jpg")
    if not os.path.exists(banner_path):
        raise FileNotFoundError(f"Banner not found: {banner_path}")
    banner_target_path = os.path.join(output_dir, "banner.jpg")
    shutil.copy(banner_path, banner_target_path)


def preprocess_only(config_paths: list[str]):

    # Load the configuration.
    config = load_configs(config_paths)
    validate_config(config)

    # Initialize the accelerator.
    accelerator = Accelerator()

    _ = preprocess(config, accelerator, ask_for_overwrite=True)


def preprocess(config, accelerator=None, ask_for_overwrite=False):
    """
    Preprocess the dataset and tokenizer. Only the main process should perform this task.
    
    Args:
        config (OmegaConf): The configuration object.
        accelerator (Accelerator): The Accelerator instance.
    
    Returns:
        datasets.DatasetDict: The tokenized datasets.
        PreTrainedTokenizerFast: The tokenizer.
    """

    # Load the dataset.
    hugging_face_id = config.dataset.hugging_face_id
    model_name = config.training.model_name
    preprocessed_path = f"./preprocessed/{model_name}"
    data_path = f"./preprocessed/{model_name}/data"
    tokenizer_path = f"./preprocessed/{model_name}/tokenizer"
    tokenized_data_path = f"./preprocessed/{model_name}/tokenized_datasets"
    checksum_path = f"./preprocessed/{model_name}/checksum.txt"

    # Compute the checksum from the configuration.
    checksum = compute_checksum_from_config(config)

    # Compare the checksum. If the checksum is different, delete the preprocessed data.
    preprocess_anyway = False
    if os.path.exists(checksum_path) and os.path.exists(data_path):
        with open(checksum_path, "r") as f:
            checksum_from_file = f.read()
        if checksum_from_file != checksum:
            accelerator.print("Checksum mismatch. Preprocessing anyway...")
            #shutil.rmtree(preprocessed_path)
            preprocess_anyway = True

    # If tokenizer and tokenized datasets exist, and ask_for_overwrite is True, ask for overwrite.
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path) and ask_for_overwrite and not preprocess_anyway:
        overwrite = input("Preprocessed data already exists. Overwrite? [y/n]: ")
        if overwrite.lower() == "y":
            accelerator.print("Deleting existing preprocessed data...")
            shutil.rmtree(preprocessed_path)

    # If tokenizer and tokenized datasets exist, load them.
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path) and not preprocess_anyway:
        accelerator.print("Loading preprocessed data...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        tokenized_datasets = load_from_disk(tokenized_data_path)
        return tokenized_datasets, tokenizer
    
    # Write the checksum to a file.
    os.makedirs(preprocessed_path, exist_ok=True)
    with open(checksum_path, "w") as f:
        f.write(checksum)

    # Download the dataset.
    if accelerator.is_local_main_process:
        accelerator.print(f"Loading dataset: {hugging_face_id}")
        raw_datasets = load_dataset(hugging_face_id)

        # Save the dataset to disk to be reused by other processes.
        raw_datasets.save_to_disk(data_path)
        accelerator.print("Dataset downloaded and saved.")
    else:
        # Other processes wait for the dataset to be downloaded and saved.
        while not os.path.exists(data_path):
            time.sleep(1)
        raw_datasets = load_dataset(data_path)
    
    accelerator.wait_for_everyone()

    # Tokenizer creation.
    if config.tokenizer.type in ["whitespace", "bpe"]:
        if accelerator.is_local_main_process:
            accelerator.print("Training whitespace tokenizer...")
            tokenizer = train_tokenizer(config.tokenizer, raw_datasets)
            tokenizer.save_pretrained(tokenizer_path)
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    elif config.tokenizer.type == "pretrained":
        from transformers import AutoTokenizer
        if accelerator.is_local_main_process:
            tokenizer_id = config.tokenizer.pretrained_id
            accelerator.print(f"Loading pre-trained tokenizer: {tokenizer_id}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            if tokenizer.pad_token is None and "GPT2TokenizerFast" in str(type(tokenizer)):
                tokenizer.pad_token = tokenizer.eos_token
            else:
                #tokenizer.add_tokens("[PAD]")
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                assert False, f"Tokenizer type not supported: {type(tokenizer)}"
            tokenizer.save_pretrained(tokenizer_path)
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer_config.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer.type}")
    
    accelerator.wait_for_everyone()

    # Assign the vocabulary size to the model configuration.
    #assert vocab_size > 0
    #config.model.vocab_size = vocab_size

    # Tokenize the datasets.
    def tokenize_function(example):
        tokenized_example = tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=config.model.context_length,
        ) 
        return {
            "input_ids": tokenized_example["input_ids"]
        } 

    if accelerator.is_local_main_process:
        accelerator.print("Tokenizing datasets...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function, 
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=multiprocessing.cpu_count()
        )
        tokenized_datasets.save_to_disk(tokenized_data_path)
    else:
        while not os.path.exists(tokenized_data_path):
            time.sleep(1)
        time.sleep(1)
        tokenized_datasets = load_from_disk(tokenized_data_path)

    accelerator.wait_for_everyone()

    # Check a sample.
    if accelerator.is_local_main_process:
        accelerator.print("Sample tokenized text:")
        sample = raw_datasets["train"][0]
        tokenized = tokenized_datasets["train"][0]
        assert list(tokenized.keys()) == ["input_ids"], list(tokenized.keys())
        accelerator.print(f"Original text: {sample}")
        accelerator.print(f"Tokenized text: {tokenized}")

    return tokenized_datasets, tokenizer


def compute_checksum_from_config(config):

    # Convert the configuration to a dictionary.
    config_dict = OmegaConf.to_container(config)

    # Use selective fields for the checksum.
    checksum_string = "HeliBrunna - A HuggingFace compatible xLSTM trainer by Dr. Tristan Behrens\n"
    checksum_string += "Configuration:\n"
    checksum_string += f"training bach size: {config_dict['training']['batch_size']}\n"
    checksum_string += f"dataset: {str(config_dict['dataset'])}\n"
    checksum_string += f"Have a pleasant day!\n"

    # Compute the checksum. Use MD5.
    checksum = hashlib.md5(checksum_string.encode()).hexdigest()
    return checksum


def train_tokenizer(tokenizer_config, raw_datasets):
    """
    Train a tokenizer based on the given configuration and raw datasets.
    Args:
        tokenizer_config (TokenizerConfig): The configuration for the tokenizer.
        raw_datasets (dict): A dictionary containing the raw datasets.
    Returns:
        PreTrainedTokenizerFast: The trained tokenizer.
    Raises:
        ValueError: If the tokenizer type is unknown.
    """
    
    # Initialize the tokenizer.
    if tokenizer_config.type == "whitespace":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[EOS]"]
        )
    elif tokenizer_config.type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[EOS]"]
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_config.tokenizer_type}")
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Train the tokenizer.
    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx : start_idx + 1000]
            yield samples["text"]
    training_corpus = get_training_corpus()
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    # Convert the tokenizer to a fast tokenizer.
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer_path = os.path.join(tempdir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Return the tokenizer.
    return tokenizer


if __name__ == "__main__":

    main()