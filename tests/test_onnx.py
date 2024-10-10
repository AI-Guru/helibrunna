import tempfile
import os
from safetensors.torch import save_file
import traceback
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from source.utilities import load_config, model_from_config, save_model
from source.languagemodel import LanguageModel
from source.onnxlanguagemodel import OnnxLanguageModel


def main():

    config_paths = [
        "jsfakes_garland_mamba.yaml",
        "jsfakes_garland_pharia.yaml",
        
        "jsfakes_garland_transformer.yaml",
        "jsfakes_garland_aethon.yaml",
        "jsfakes_garland_minillama.yaml",
        "jsfakes_garland_llamathree.yaml",

        "jsfakes_garland_mingru.yaml",
        "jsfakes_garland_minlstm.yaml",

        "jsfakes_garland_xlstm.yaml",
    ]

    # Test all the configs. 
    status_list = []
    for config_path in config_paths:
        try:
            test(config_path)
            status_list += [(True, config_path, "")]
        except Exception as e:
            print(f"Test failed: {e}")
            status_list += [(False, config_path, str(e))]
            traceback.print_exc()
    
    # Print the results
    print("")
    print("-" * 80)
    print("")
    print("Results:")
    for (status, config_path, exception) in status_list:
        result = "✅" if status else "❌"
        result += f" {config_path}"
        if exception != "":
            result += f" {exception}"
        print(result)
    print("")
    print("-" * 80)
    print("")
    


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
    model.eval()  # Set the model to evaluation mode

    # Test the model.
    input = torch.randint(0, 1000, (1, 10))
    output = model(input)
    del output

    # Save the model and load it as an ONNX model.
    with tempfile.TemporaryDirectory() as tempdir:

        # Save the model.
        model_path = os.path.join(tempdir, "output", "checkpoint-666-last")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model to {model_path}")
        save_model(model, model_config, model_path)

        # Load the Pytorch model and the ONNX model.
        search_path = os.path.dirname(model_path)
        pytorch_model = LanguageModel(search_path, ignore_tokenizer=True)
        pytorch_model.model.eval() 
        onnx_model = OnnxLanguageModel(search_path, quantization=False, ignore_tokenizer=True)

        # Expected output shape
        expected_output_shape = (1, 10, 1000)

        # Test the Pytorch model first.
        input = torch.randint(0, 1000, (1, 10))
        output_pytorch = pytorch_model.predict(input)
        output_pytorch_2 = pytorch_model.predict(input)
        if not expected_output_shape == output_pytorch.shape:
            raise ValueError(f"Expected output shape {expected_output_shape}, got {output_pytorch.shape}")
        if not torch.allclose(output_pytorch.to("cpu"), output_pytorch_2.to("cpu"), atol=1e-5):
            raise ValueError("Output is not the same for the Pytorch model")

        # Test the ONNX model.
        output_onnx = onnx_model.predict(input)
        output_onnx_2 = onnx_model.predict(input)
        if not expected_output_shape == output_onnx.shape:
            raise ValueError(f"Expected output shape {expected_output_shape}, got {output_onnx.shape}")
        if not torch.allclose(output_onnx.to("cpu"), output_onnx_2.to("cpu"), atol=1e-5):
            raise ValueError("Output is not the same for the ONNX model")

        # Output should be the same for both models.
        if not torch.allclose(output_pytorch.to("cpu"), output_onnx.to("cpu"), atol=1e-5):
            raise ValueError("Output is not the same")

        # Done.
        del onnx_model


if __name__ == "__main__":
    main()