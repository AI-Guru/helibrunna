"""
Helibrunna - A HuggingFace compatible xLSTM trainer.

Copyright (c) 2024 Dr. Tristan Behrens

All rights reserved. This software and associated documentation files (the "Software") may only be used, copied, modified, merged, published, distributed, sublicensed, and/or sold under the terms and conditions set forth by the author, Dr. Tristan Behrens.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import fire
import os
import glob
from omegaconf import OmegaConf
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from transformers import PreTrainedTokenizerFast
from dacite import from_dict
import torch
from safetensors.torch import load_file
import time
from source.utilities import display_logo


def generate(
        model_path_or_repo: str,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = 100,
) -> None:
    """
    Generates text continuation based on a given prompt using a pre-trained language model.
    Args:
        model_path_or_repo (str): The path to the model or the Hugging Face repository ID.
        prompt (str): The prompt text to generate continuation from.
        temperature (float, optional): The temperature value for sampling from the distribution. Defaults to 0.5.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
    Raises:
        ValueError: If the model weights, tokenizer, or config are not found at the specified paths.
    Returns:
        None
    """

    # Display the logo.
    display_logo()

    # Set the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download the model if it doesn't exist. Or at least try to.
    if not os.path.exists(model_path_or_repo):
        from huggingface_hub import snapshot_download
        try:
            model_path=snapshot_download(repo_id=model_path_or_repo)
            tokenizer_path=model_path
        except Exception as e:
            raise f"Failed to download the model: {e}"
    
    # Use a local model.
    else:

        # Set the model path and tokenizer path.
        model_path = None
        tokenizer_path = model_path_or_repo

        # Find all the checkpoint folders, folders that start with "checkpoint-". Then find the last one.
        checkpoint_folders = glob.glob(os.path.join(model_path_or_repo, "checkpoint-*"))
        for checkpoint_folder in checkpoint_folders:
            if checkpoint_folder.endswith("-last"):
                model_path = checkpoint_folder
                break
        if model_path is None:
            raise ValueError("No model checkpoint found.")

        # Find the tokenizer folder.
        if os.path.exists(os.path.join(model_path_or_repo, "tokenizer.json")):
            tokenizer_path = model_path_or_repo
        if not os.path.exists(tokenizer_path):
            raise ValueError("Tokenizer not found.")

    # Load the config.
    print(f"Loading model config from {model_path}...")
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        raise ValueError(f"Config not found at {config_path}")
    model_config = OmegaConf.load(config_path)
    print(model_config)

    # Create the model from the config.
    print("Creating model...")
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_config))).to(device)

    # Load the weights from the checkpoint.
    print("Loading model weights...")
    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weights_path):
        raise ValueError(f"Weights not found at {weights_path}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    # Load the tokenizer.
    print("Loading tokenizer...")
    tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Tokenize the prompt.
    print("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    assert inputs.shape[0] == 1

    # Generate the continuation.
    start_time = time.time()
    tokens_count = 0
    while inputs.shape[1] < max_length:

        # Generate the continuation.
        outputs = model(inputs.to(device=device))
        assert outputs.shape[0] == 1

        # Use the temperature to sample from the distribution.
        outputs = outputs / temperature
        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        outputs = torch.multinomial(outputs[0, -1], num_samples=1)

        # Add to the inputs.
        inputs = torch.cat([inputs, outputs.unsqueeze(0)], dim=1)
        
        # Increment the tokens count.
        tokens_count += 1

    # Print the elapsed time and tokens per second.
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s")
    tokens_per_second = tokens_count / elapsed_time
    print(f"Tokens per second: {tokens_per_second:.2f}")

    # Decode the output.
    output = tokenizer.decode(inputs[0].tolist())
    print(output)


# Entry point.
if __name__ == "__main__":
    fire.Fire(generate)