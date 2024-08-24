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

import fire
import json
from source.languagemodel import LanguageModel


def generate(
        model_path_or_repo: str,
        prompt: str,
        temperature: float = 1.0,
        max_length: int = 100,
) -> None:
    """
    Generates text continuation based on a given prompt using a pre-trained language model.
    Args:
        model_path_or_repo (str): The path to the model or the Hugging Face repository ID.
        prompt (str): The prompt text to generate continuation from.
        temperature (float, optional): The temperature value for sampling from the distribution. Defaults to 0.5.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
    Raises:
        ValueError: If the model weights, tokenizer, or config are not found at the specified paths.
    Returns:
        None
    """

    # Load the model.
    print("Loading the model...")
    model = LanguageModel(model_path_or_repo)

    # Generate some text.
    print("Generating text...")
    output = model.generate(
        prompt=prompt,
        temperature=temperature,
        max_length=max_length,
        return_structured_output=True
    )
    print(json.dumps(output, indent=4))


# Entry point.
if __name__ == "__main__":
    fire.Fire(generate)