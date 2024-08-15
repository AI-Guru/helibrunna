import fire
import os
from omegaconf import OmegaConf
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from dacite import from_dict
import torch
from safetensors.torch import load_file


def generate(
        model_path: str,
        temperature: float,
        max_length: int,
        prompt: str
):
    # Load the config.
    print(f"Loading model config from {model_path}...")
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        raise ValueError(f"Config not found at {config_path}")
    model_config = OmegaConf.load(config_path)
    print(model_config)

    # Create the model from the config.
    print("Creating model...")
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_config))).to(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the weights from the checkpoint.
    print("Loading model weights...")
    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weights_path):
        raise ValueError(f"Weights not found at {weights_path}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    # Generate text.


if __name__ == "__main__":
    fire.Fire(generate)