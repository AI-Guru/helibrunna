import os
import colorama
from omegaconf import DictConfig, OmegaConf
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
        "lr_warmup_steps": int,
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
    },
}

def validate_config(config: DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    validate_dict(config, config_schema, parent_key='')


def human_readable_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"  # For numbers larger than 1T