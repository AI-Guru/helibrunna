from omegaconf import OmegaConf
from dacite import from_dict
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
import sys
sys.path.append(".")
from source.utilities import model_from_config

# Load the config.
config_string = """ 
model:
  vocab_size: 50257
  num_blocks: 24
  embedding_dim: 384
  mlstm_block:
    mlstm:
      num_heads: 4
  slstm_block:
    slstm:
      num_heads: 4
  slstm_at: [3, 20]
  context_length: 2048
"""

if len(sys.argv) > 1:
  config_path = sys.argv[1]
  config = OmegaConf.load(config_path)
else:
  config = OmegaConf.create(config_string)

# Create the model.
model = model_from_config(config.model)
print(model)

# Get the number of parameters.
number_of_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {number_of_parameters:_}")