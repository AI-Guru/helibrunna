import fire
import os
from omegaconf import OmegaConf
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from transformers import PreTrainedTokenizerFast
from dacite import from_dict
import torch
from safetensors.torch import load_file
import time

def generate(
        model_path: str,
        tokenizer_path: str,
        temperature: float,
        max_length: int,
        prompt: str
):
    
    # Set the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        outputs = torch.multinomial(outputs[0, 0], num_samples=1)

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

if __name__ == "__main__":
    fire.Fire(generate)