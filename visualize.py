# Script that renders an xLSTM model with torchviz
import fire
import torch
from torchviz import make_dot
from omegaconf import OmegaConf
from dacite import from_dict
from source.utilities import load_config, human_readable_number
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def visualize(config_path):
    config = load_config(config_path)
    config.model.vocab_size = 502

    model_config = from_dict(xLSTMLMModelConfig, OmegaConf.to_container(config.model))
    model = xLSTMLMModel(model_config).to("cuda")

    # Print the number of parameters.
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {number_of_parameters:_}")

    # Make human readable.
    number_of_parameters_human = human_readable_number(number_of_parameters)
    print(f"Number of parameters: {number_of_parameters_human}")

    x = torch.randint(0, config.model.vocab_size, (1, config.model.context_length)).to("cuda")
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render("xLSTMModel", format="png", cleanup=True)


if __name__ == "__main__":
    fire.Fire(visualize)