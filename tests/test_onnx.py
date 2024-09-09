import tempfile
import os
from safetensors.torch import save_file
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from source.utilities import load_config, model_from_config, save_model
from source.languagemodel import LanguageModel
from source.onnxlanguagemodel import OnnxLanguageModel


def main():

    config_paths = [
        #"jsfakes_garland_mamba.yaml",
        #"jsfakes_garland_pharia.yaml",
        #"jsfakes_garland_transformer.yaml",
        "jsfakes_garland_aethon.yaml",
        #"jsfakes_garland_xlstm.yaml",
    ]

    # Test all the configs. 
    status_list = []
    for config_path in config_paths:
        try:
            test(config_path)
            status_list.append(True)
        except Exception as e:
            print(f"Test failed: {e}")
            status_list.append(False)
            traceback.print_exc()
    
    # Print the results
    print("")
    print("Results:")
    for status, config_path in zip(status_list, config_paths):
        result = "✅" if status else "❌"
        print(f"  {result} {config_path}")


def test(config_path):
    full_config_path = f"configs/{config_path}"
    assert os.path.exists(full_config_path), f"Config file {full_config_path} does not exist"

    # Load the config file.
    config = load_config(full_config_path)
    model_config = config["model"]
    model_config["vocab_size"] = 1000
    print(model_config)

    # Create the model.
    model = model_from_config(model_config, device="cpu")

    with tempfile.TemporaryDirectory() as tempdir:
        # Save the model.
        model_path = os.path.join(tempdir, "output", "checkpoint-666-last")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model to {model_path}")
        save_model(model, model_config, model_path)
        print("Test passed")

        # Print the tree of the path.
        print(f"Tree of {model_path}")
        for root, dirs, files in os.walk(model_path):
            level = root.replace(model_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

        # Create the ONNX model.
        search_path = os.path.dirname(model_path)
        onnx_model = OnnxLanguageModel(search_path, quantization=False, ignore_tokenizer=True)


if __name__ == "__main__":
    main()