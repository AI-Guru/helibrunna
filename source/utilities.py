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
 
import os
import colorama
from omegaconf import DictConfig, OmegaConf
import torch
from typing import List, Tuple, Dict
from dacite import from_dict
from collections.abc import MutableMapping
import sys


def display_logo():
    """
    Display the logo by printing it line by line with a cyberpunk color scheme.

    Raises:
        FileNotFoundError: If the logo file is missing.
    """

    # Get the path of this script and use it to find the logo.
    script_path = os.path.dirname(os.path.realpath(__file__))
    search_path = os.path.dirname(script_path)

    # Load the logo.
    logo_path = os.path.join(search_path, "assets", "asciilogo.txt")
    if not os.path.exists(logo_path):
        raise FileNotFoundError("The logo file is missing.")
    with open(logo_path, "r") as f:
        logo = f.read()

    # Print the logo line by line. Use colorama to colorize the output. Use a cyberpunk color scheme.
    for line_index, line in enumerate(logo.split("\n")):
        color = colorama.Fore.GREEN
        style = colorama.Style.BRIGHT if line_index % 2 == 0 else colorama.Style.NORMAL
        print(color + style + line)
    print(colorama.Style.RESET_ALL)


def validate_dict(config: DictConfig, schema: dict, parent_key: str = ''):
    for key, expected_type in schema.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if key not in config:
            raise ValueError(f"Missing key: {full_key}")
        if isinstance(expected_type, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"Expected {full_key} to be a dict, got {type(config[key])}")
            validate_dict(config[key], expected_type, full_key)
        elif isinstance(expected_type, list):
            if not isinstance(config[key], list):
                raise ValueError(f"Expected {full_key} to be a list, got {type(config[key])}")
            for item in config[key]:
                if not isinstance(item, expected_type[0]):
                    raise ValueError(f"Expected items in {full_key} to be of type {expected_type[0]}, got {type(item)}")
        else:
            if not isinstance(config[key], expected_type):
                raise ValueError(f"Expected {full_key} to be of type {expected_type}, got {type(config[key])}")

# Define the schema
config_schema = {
    "training": {
        "model_name": str,
        "batch_size": int,
        "lr": float,
        "lr_warmup_steps": (int, str),
        "lr_decay_until_steps": (int, str),
        "lr_decay_factor": float,
        "weight_decay": float,
        "amp_precision": str,
        "weight_precision": str,
        "enable_mixed_precision": bool,
        "num_epochs": int,
        "output_dir": str,
        "save_every_step": int,
        "log_every_step": int,
        "wandb_project": str,
        "torch_compile": bool,
    },
    "model": {
        "num_blocks": int,
        "embedding_dim": int,
        "mlstm_block": {
            "mlstm": {
                "num_heads": int,
            },
        },
        "slstm_block": {
            "slstm": {
                "num_heads": int,
            },
        },
        "slstm_at": [int],
        "context_length": int,
    },
    "dataset": {
        "hugging_face_id": str,
    },
    "tokenizer": {
        "type": str,
        "fill_token": str,
    },
}


def load_config(config_path: str) -> OmegaConf:
    """
    Load the configuration from the specified path.
    Args:
        config_path (str): The path to the configuration file.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    Returns:
        OmegaConf: The configuration object.
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_yaml = f.read()
    config = OmegaConf.create(config_yaml)
    OmegaConf.resolve(config)
    return config


def load_configs(config_paths: str) -> OmegaConf:
    """
    Load and merge configurations from the specified paths.
    Args:
        config_paths (str): The paths to the configuration files.

    Raises:
        FileNotFoundError: If any of the configuration files are not found.
    Returns:
        OmegaConf: The merged configuration object.
    """
    
    merged_config = OmegaConf.create()

    for config_path in config_paths:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_yaml = f.read()
        
        new_config = OmegaConf.create(config_yaml)
        OmegaConf.resolve(new_config)

        # Detect and warn about overwrites
        for key, value in new_config.items():
            if key in merged_config:
                if isinstance(value, DictConfig) and isinstance(merged_config[key], DictConfig):
                    # Merge nested configurations
                    OmegaConf.merge(merged_config[key], value)
                else:
                    print(f"Warning: Overwriting '{key}' from '{merged_config[key]}' to '{value}' in config file: {config_path}")
            merged_config[key] = value
    
    return merged_config


def validate_config(config: DictConfig):
    """
    Validate the configuration dictionary.
    Args:
        config (DictConfig): The configuration dictionary to validate.
    Returns:
        None
    Raises:
        ValidationError: If the configuration is invalid.
    """

    config = OmegaConf.to_container(config, resolve=True)
    validate_dict(config, config_schema, parent_key='')


def human_readable_number(num):
    """
    Converts a number into a human-readable format.

    Args:
        num (float): The number to be converted.

    Returns:
        str: The human-readable representation of the number.

    Examples:
        >>> human_readable_number(1000)
        '1.0K'
        >>> human_readable_number(1500000)
        '1.5M'
        >>> human_readable_number(5000000000)
        '5.0B'
    """

    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"  # For numbers larger than 1T


def is_torch_compile_ready():
    """
    Check if the current system is ready for compiling Torch code.
    Returns:
        bool: True if the system is ready for compiling Torch code, False otherwise.
    """

    # If there is no GPU, return False.  
    if not torch.cuda.is_available():
        return False
    
    # Check the device capability.
    device_capability = torch.cuda.get_device_capability()
    if device_capability in ((7, 0), (8, 0), (9, 0)):
        return True
    
    return False


def model_from_config(model_config: DictConfig, device:str) -> torch.nn.Module:
    """
    Create a model based on the provided model configuration.

    Args:
        model_config (DictConfig): The configuration for the model.

    Returns:
        The created model.

    Raises:
        ValueError: If the model type is unknown.
    """
    
    # Get the model type from the configuration.
    model_type = model_config.get("type", "xLSTMLMModel")
    
    # Create the xLSTMLMModel.
    if model_type == "xLSTMLMModel":
        print("Creating xLSTMLMModel...")
        from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
        
        # If there is no GPU, use the vanilla backend.
        if not torch.cuda.is_available():
            model_config.backend = "vanilla"
            model_config.slstm_block.slstm.backend = "vanilla"
            model_config.mlstm_block.mlstm.backend = "vanilla"
        model_config_object = from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_config))
        
        # Create the model.
        model = xLSTMLMModel(model_config_object)
        model.reset_parameters()
    
    # Create the GPT2LMModel.
    elif model_type == "gpt2":
        print("Creating GPT2LMModel...")
        from .models.gpttwo import GPT2LMModel, GPT2LMModelConfig
        model_config_object = from_dict(GPT2LMModelConfig, OmegaConf.to_container(model_config))
        model = GPT2LMModel(model_config_object)
    
    # Create the MambaLM.
    elif model_type == "mamba":
        print("Creating Mamba LM...")
        from mambapy.lm import LM, MambaConfig
        model_config_object = from_dict(MambaConfig, OmegaConf.to_container(model_config))
        model = LM(model_config_object, model_config.vocab_size)
    
    # Create the Transformer.
    elif model_type == "transformer":
        from .models.transformer import TransformerConfig, Transformer
        model_config_object = from_dict(TransformerConfig, OmegaConf.to_container(model_config))
        model = Transformer(model_config_object)
    
    # Create a Pharia instance.
    elif model_type == "pharia":
        from .models.pharia import PhariaConfig, PhariaModel
        model_config_object = from_dict(PhariaConfig, OmegaConf.to_container(model_config))
        model = PhariaModel(model_config_object)
    
    # Create a TransformerXL instance.
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move the model to the device.
    model.to(device)
    return model
