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

import torch
from .languagemodel import LanguageModel
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
import numpy as np


class OnnxLanguageModel(LanguageModel): 


    def __init__(self, model_path: str, quantization=False, **kwargs):
        super().__init__(model_path, **kwargs)
        assert self.model is not None, "Model not loaded"

        # Move the model to the CPU.
        self.model.to("cpu")
        self.model.eval()  # Set the model to evaluation mode

        # Define example input tensors for export
        batch_size = 1
        seq_length = self.config.context_length
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))

        # Prepare dynamic_axes to allow dynamic batch size and sequence length
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'seq_length'},
            'output': {0: 'batch_size', 1: 'seq_len'}
        }

        # Export the model to ONNX
        torch.onnx.export(
            self.model,  # Model
            (input_ids, ),  # Model inputs
            "model.onnx",  # Output ONNX file
            input_names=['input_ids'],  # Input names
            output_names=['output'],  # Output name
            dynamic_axes=dynamic_axes,  # Dynamic axes for variable batch size and seq length
            opset_version=16,  # ONNX opset version
            do_constant_folding=True,  # Whether to apply constant folding for optimization
            verbose=True  # Print the export process
        )

        # Quantize the model.
        if quantization:
            weight_type = QuantType.QUInt8
            _ = quantize_dynamic("model.onnx", "model.onnx", weight_type=weight_type)

        # Unload the model.
        del self.model

        # Create an ONNX Runtime Inference Session
        ort_session = onnxruntime.InferenceSession("model.onnx")

        # Run a test.
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        input_ids_np = input_ids.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: input_ids_np}
        ort_outs = ort_session.run(None, ort_inputs)

        # Run another test.
        input_ids = torch.randint(0, self.config.vocab_size, (1, 1))
        input_ids_np = input_ids.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: input_ids_np}
        ort_outs = ort_session.run(None, ort_inputs)

        self.onnx_session = ort_session



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
        Generates a continuation for a given prompt using the ONNX language model.
        """
        # Tokenize the prompt.
        inputs = self.tokenizer(prompt, return_tensors="np").input_ids
        assert inputs.shape[0] == 1  # Ensure batch size is 1

        # Determine the end token ids.
        end_token_ids = []
        for end_token in end_tokens:
            assert end_token in self.tokenizer.vocab
            end_token_ids.append(self.tokenizer.convert_tokens_to_ids(end_token))

        # Initialize the ids to mask.
        ids_to_mask = []

        # Mask the forbidden tokens.
        for forbidden_token in forbidden_tokens:
            assert forbidden_token in self.tokenizer.vocab
            forbidden_token_id = self.tokenizer.convert_tokens_to_ids(forbidden_token)
            ids_to_mask += [forbidden_token_id]

        # Generate the continuation.
        start_time = time.time()
        tokens_count = 0
        while inputs.shape[1] < max_length:
            # Stop if the maximum context length is reached.
            if inputs.shape[1] >= self.config.context_length:
                print("Warning: The maximum context length has been reached.")
                break

            # Run ONNX inference
            ort_inputs = {self.onnx_session.get_inputs()[0].name: inputs}
            ort_outs = self.onnx_session.run(None, ort_inputs)

            # Convert the ONNX output (logits) to numpy
            logits = ort_outs[0]
            assert logits.shape[0] == 1  # Ensure batch size is 1

            # Mask special tokens (e.g., padding, unk, etc.)
            logits[:, :, self.tokenizer.all_special_ids] = float("-inf")

            # Apply temperature scaling
            logits = logits / temperature

            # Convert logits to probabilities using softmax
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

            # Sample from the probabilities distribution
            sampled_token = np.random.choice(np.arange(probs.shape[-1]), p=probs[0, -1])

            # Add the sampled token to inputs
            inputs = np.concatenate([inputs, np.array([[sampled_token]])], axis=1)

            # Increment the tokens count
            tokens_count += 1

            # Check if the end token is reached
            if sampled_token in end_token_ids:
                break

        # Print the elapsed time and tokens per second
        elapsed_time = time.time() - start_time
        tokens_per_second = tokens_count / elapsed_time

        # Decode the output
        output = self.tokenizer.decode(inputs[0].tolist())

        # Return the output
        if not return_structured_output:
            return output
        else:
            # Return structured output
            return {
                "output": output,
                "elapsed_time": elapsed_time,
                "tokens_per_second": tokens_per_second
            }
        
    def predict(self, input: torch.Tensor):
        """
        Predicts the output for a given input using the ONNX language model.
        """
        assert input.shape[0] == 1
        input_np = input.cpu().numpy()
        ort_inputs = {self.onnx_session.get_inputs()[0].name: input_np}
        ort_outs = self.onnx_session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0])
    
    

    def summary(self):
        print("OnnxLanguageModel")

    