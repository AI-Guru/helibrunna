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


def display_logo():

    # Load the logo.
    logo_path = os.path.join("assets", "asciilogo.txt")
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
    config = OmegaConf.to_container(config, resolve=True)
    validate_dict(config, config_schema, parent_key='')


def human_readable_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"  # For numbers larger than 1T

def is_torch_compile_ready():

    # If there is no GPU, return False.  
    if not torch.cuda.is_available():
        return False
    
    # Check the device capability.
    device_capability = torch.cuda.get_device_capability()
    if device_capability in ((7, 0), (8, 0), (9, 0)):
        return True
    
    return False
