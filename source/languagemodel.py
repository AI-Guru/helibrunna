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
import glob
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast
import torch
from safetensors.torch import load_file
import time
from .utilities import display_logo, model_from_config


class LanguageModel:

    def __init__(self, model_path_or_repo, config_overrides={}, mask_special_tokens=True, device="auto"):
        """
        Initializes the LanguageModel object.
        Args:
            model_path_or_repo (str): The path to the model or the repository ID.
        Raises:
            ValueError: If the model checkpoint, tokenizer, config, or weights are not found.
            Exception: If failed to download the model.
        Returns:
            None
        """

        # Set the maskt_special_tokens flag.
        self.mask_special_tokens = mask_special_tokens

        # Set the device. CPU is default.
        if device != "auto":
            
            # Check if CUDA is available.
            if not torch.cuda.is_available() and device == "cuda":
                raise ValueError("CUDA is not available on this system.")
            
            # Check if MPS is available.
            if not torch.backends.mps.is_available() and device == "mps":
                raise ValueError("MPS is not available on this system.")

            # Set the device.
            self.device = device

        # Set the device to auto.
        else:     

            # Set the device to CPU if auto is selected.
            self.device = "cpu" if device == "auto" else device

            # Check if CUDA is available.
            if torch.cuda.is_available() and device == "auto":
                self.device = "cuda"

            # See if MPS is available.
            # Note: This is disabled for now. It's not working as expected. It is very slow.
            #if torch.backends.mps.is_available():
            #    self.device = "mps"

        # Display the logo.
        display_logo()

        # Download the model if it doesn't exist. Or at least try to.
        if not os.path.exists(model_path_or_repo):
            from huggingface_hub import snapshot_download
            try:
                model_path=snapshot_download(repo_id=model_path_or_repo)
                tokenizer_path=model_path
            except Exception as e:
                raise f"Failed to download the model: {e}"
        
        # Use a local model.
        else:
            # Set the model path and tokenizer path.
            model_path = None
            tokenizer_path = model_path_or_repo

            # Find all the checkpoint folders, folders that start with "checkpoint-". Then find the last one.
            checkpoint_folders = glob.glob(os.path.join(model_path_or_repo, "checkpoint-*"))
            for checkpoint_folder in checkpoint_folders:
                if checkpoint_folder.endswith("-last"):
                    model_path = checkpoint_folder
                    break
            if model_path is None:
                raise ValueError("No model checkpoint found.")

            # Find the tokenizer folder.
            if os.path.exists(os.path.join(model_path_or_repo, "tokenizer.json")):
                tokenizer_path = model_path_or_repo
            if not os.path.exists(tokenizer_path):
                raise ValueError("Tokenizer not found.")

        # Load the config.
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}")
        model_config = OmegaConf.load(config_path)

        # Override the config.
        if config_overrides != {} and config_overrides is not None:
            model_config = OmegaConf.merge(model_config, config_overrides)
        import json
        print(json.dumps(OmegaConf.to_container(model_config), indent=4))

        # Create the model from the config.
        model = model_from_config(model_config, device=self.device)
        model.to(self.device)
        self.config = model_config

        # Load the weights from the checkpoint.
        weights_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights not found at {weights_path}")
        state_dict = load_file(weights_path)

        # TODO: Permute the last two dimensions of these parameters: xlstm_block_stack.blocks.2.xlstm.slstm_cell._recurrent_kernel_:
        # Check if we have an xLSTM model and if CUDA is not available.
        if not torch.cuda.is_available() and model_config.get("type", "xLSTMLMModel") == "xLSTMLMModel":
            print(state_dict.keys())
            endings = ["xlstm.slstm_cell._recurrent_kernel_"]
            for key, values in state_dict.items():
                for ending in endings:
                    if key.endswith(ending):
                        print(key)
                        print(values.shape)
                        
                        # Option: Permute the last two dimensions.
                        values = values.permute(0, 2, 1)
                        
                        # Option: View the tensor.
                        #new_shape = (values.shape[0], values.shape[2], values.shape[1])
                        #values = values.view(new_shape)
                        
                        print(values.shape)
                        state_dict[key] = values
                        break

        # Load the weights into the model.
        model.load_state_dict(state_dict)
        self.model = model

        # Load the tokenizer.
        tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer


    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = 100,
        end_tokens: list[str] = [],
        forbidden_tokens: list[str] = [],
        return_structured_output: bool = False
    ):
        """
        Generates a continuation for a given prompt using the language model.
        Args:
            prompt (str): The prompt to generate a continuation for.
            temperature (float, optional): The temperature value for controlling the randomness of the generated output. 
                Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.5) make it more deterministic. 
                Defaults to 1.0.
            max_length (int, optional): The maximum length of the generated output. Defaults to 100.
            end_tokens (list[str], optional): A list of end tokens that, if encountered, will stop the generation process. 
                Defaults to an empty list.
            return_structured_output (bool, optional): If True, returns a dictionary with the generated output, elapsed time, 
                and tokens per second. If False, returns only the generated output as a string. Defaults to False.
        Returns:
            str or dict: The generated output as a string if return_structured_output is False. 
                A dictionary with the generated output, elapsed time, and tokens per second if return_structured_output is True.
        """    

        # Tokenize the prompt.
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert inputs.shape[0] == 1

        # Determine the end tokens ids.
        end_token_ids = []
        for end_token in end_tokens:
            assert end_token in self.tokenizer.vocab
            end_token_ids.append(self.tokenizer(end_token).input_ids[0])

        # Initialize the ids to mask.
        ids_to_mask = []

        # Mask the forbidden tokens.
        for forbidden_token in forbidden_tokens:
            assert forbidden_token in self.tokenizer.vocab
            ids_to_mask.extend(self.tokenizer(forbidden_token).input_ids)

        # Generate the continuation.
        start_time = time.time()
        tokens_count = 0
        while inputs.shape[1] < max_length:

            # Stop if the maximum context length is reached.
            if inputs.shape[1] >= self.config.context_length:
                print("Warning: The maximum context length has been reached.")
                break

            # Generate the continuation.
            outputs = self.model(inputs.to(device=self.device))
            assert outputs.shape[0] == 1

            # Mask the tokens.
            outputs[:, :, self.tokenizer.all_special_ids] = float("-inf")

            # Use the temperature to sample from the distribution.
            outputs = outputs / temperature
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            outputs = torch.multinomial(outputs[0, -1], num_samples=1)

            # Add to the inputs.
            inputs = torch.cat([inputs, outputs.unsqueeze(0)], dim=1)
            
            # Increment the tokens count.
            tokens_count += 1

            # Check if the end token is reached.
            if outputs[0] in end_token_ids:
                break

        # Print the elapsed time and tokens per second.
        elapsed_time = time.time() - start_time
        tokens_per_second = tokens_count / elapsed_time

        # Decode the output.
        output = self.tokenizer.decode(inputs[0].tolist())

        # Return the output.
        if not return_structured_output:
            return output
        
        # Return the structured output.
        else:
            return {
                "output": output,
                "elapsed_time": elapsed_time,
                "tokens_per_second": tokens_per_second
            }
        
    def summary(self):
        """
        Prints a summary of the model. Makes the model architecture readable. Includes the number of parameters.
        """

        # Print the model.
        print(self.model)

        # Get the number of parameters.
        number_of_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {number_of_parameters:_}")
        sizes = ["", "K", "M", "B", "T"]
        size_index = 0
        while number_of_parameters > 1000:
            number_of_parameters /= 1000
            size_index += 1
        print(f"Number of parameters: {number_of_parameters:.2f}{sizes[size_index]}")
        
        # Size of the model.
        # Get the total size of all the markdown files. And make it human readable.
        number_of_parameters = sum(p.numel() for p in self.model.parameters())
        total_size = number_of_parameters * 4
        sizes = ["B", "KB", "MB", "GB", "TB"]
        size_index = 0
        while total_size > 1024:
            total_size /= 1024
            size_index += 1
        print(f"Total size of the model: {total_size:.2f}{sizes[size_index]} for precision 32-bit floats.")

        # Print on which device the model is running. 
        print(f"Device: {self.device}")
