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
import sys
sys.path.append(".")
from source.utilities import display_logo, human_readable_number, load_configs, validate_config, is_torch_compile_ready, model_from_config


config_paths = ["configs/jsfakes_garland_pharia.yaml"]

config = load_configs(config_paths)
config.model.vocab_size = 111
config.model.pad_token_id = 1
print(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_from_config(config.model, device)
print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {human_readable_number(num_params)}")

# Test inference.
batch_size = 1

input = torch.randint(0, config.model.vocab_size, (batch_size, 100)).to(device)
print(f"Input shape: {input.shape}")
output = model(input)
print(f"Output shape: {output.shape}")