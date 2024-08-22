

# Get the path of this file.
import os
import glob

# Go up one directory from the current file.
path = os.path.dirname(__file__)
path = os.path.abspath(path)
path = os.path.dirname(path)

import sys
sys.path.append(path)

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from dacite import from_dict
from omegaconf import OmegaConf

# Import the utilities module.
from source.utilities import load_config, human_readable_number

# Find the configs directory.
configs_path = os.path.join(path, "configs")
if not os.path.exists(configs_path):
    raise FileNotFoundError(f"Config directory not found: {configs_path}")

# Find all the config files.
config_files = glob.glob(os.path.join(configs_path, "xlstm_*.yaml"))
config_files = sorted(config_files)
for i, config_file in enumerate(config_files):
    print(f"{i+1}. {os.path.basename(config_file)}")

# Load the configs.
statistics = {}
for config_file in config_files:

    # Load the config file.
    print(f"Loading config file: {os.path.basename(config_file)}")
    config = load_config(config_file)

    model_name = os.path.basename(config_file).replace("xlstm_", "").replace(".yaml", "")
    statistics[model_name] = {}

    # Set the vocabulary size. Use the size from the GPT2 tokenizer.
    config.model.vocab_size = 50257

    # Create the model.
    print("Creating model...")
    model_config = from_dict(xLSTMLMModelConfig, OmegaConf.to_container(config.model))
    model = xLSTMLMModel(model_config)
    print(model_config)
    print(model)

    # Get the number of parameters.
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    num_params_human = human_readable_number(num_params)

    # Get the number of parameters without the embedding layer and last layer.
    embedding_layer_size = model.token_embedding.weight.numel()
    last_layer_size = model.lm_head.weight.numel()
    num_params_blocks = num_params - embedding_layer_size - last_layer_size
    num_params_blocks_human = human_readable_number(num_params_blocks)

    # Get the number of bytes.
    num_bytes = num_params * 4
    num_bytes_human = human_readable_number(num_bytes) + "B"
    num_bytes_human = num_bytes_human.replace("BB", "GB")

    # Save the statistics.
    statistics[model_name]["num_params"] = num_params
    statistics[model_name]["num_params_human"] = num_params_human
    statistics[model_name]["num_params_blocks"] = num_params_blocks
    statistics[model_name]["num_params_blocks_human"] = num_params_blocks_human
    statistics[model_name]["num_bytes"] = num_bytes
    statistics[model_name]["num_bytes_human"] = num_bytes_human


    #break

# Print the statistics.
print("\n\n")
for model_name, stats in statistics.items():
    print(f"Model: {model_name}")
    print(f"  Number of Parameters:          {stats['num_params']:_} ({stats['num_params_human']})")
    print(f"  Number of Parameters (Blocks): {stats['num_params_blocks']:_} ({stats['num_params_blocks_human']})")
    print(f"  Number of Bytes:               {stats['num_bytes']:_} ({stats['num_bytes_human']})")
    print()