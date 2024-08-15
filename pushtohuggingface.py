"""
Helibrunna - A HuggingFace compatible xLSTM trainer.

Copyright (c) 2024 Dr. Tristan Behrens

All rights reserved. This software and associated documentation files (the "Software") may only be used, copied, modified, merged, published, distributed, sublicensed, and/or sold under the terms and conditions set forth by the author, Dr. Tristan Behrens.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import fire
import os
import glob
from huggingface_hub import create_repo
from huggingface_hub import HfApi

def push_to_huggingface(
        model_path: str,
        username_or_orga: str,
        repo_name: str,
        private: bool = False,
):
    """
    Pushes a model to Hugging Face Model Hub repository.
    Args:
        model_path (str): The path to the model directory.
        username_or_orga (str): The username or organization name on Hugging Face Model Hub.
        repo_name (str): The name of the repository on Hugging Face Model Hub.
        private (bool, optional): Whether the repository should be private. Defaults to False.
    Raises:
        ValueError: If the model path does not exist or if no checkpoints are found.
    """

    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")
    
    # Add the files that are required.
    files = [
        os.path.join(model_path, "tokenizer.json"),
        os.path.join(model_path, "special_tokens_map.json"),
        os.path.join(model_path, "tokenizer_config.json"),
        os.path.join(model_path, "README.md"),
        os.path.join(model_path, "banner.jpg"),
    ]

    # Find all the checkpoints. Make sure these are folders.
    checkpoints = glob.glob(os.path.join(model_path, "checkpoint-*"))
    checkpoints = [checkpoint for checkpoint in checkpoints if os.path.isdir(checkpoint)]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoints found.")

    # Find the last checkpoint. It begins with "checkpoint-" and ends with a "-last".
    last_checkpoint = None
    for checkpoint in checkpoints:
        if checkpoint.endswith("-last"):
            last_checkpoint = checkpoint
            break
    if last_checkpoint is None:
        raise ValueError("No last checkpoint found.")
    files += [os.path.join(last_checkpoint, "model.safetensors")]
    files += [os.path.join(last_checkpoint, "config.yaml")]

    # Check if all files exist.
    for file in files:
        if not os.path.exists(file):
            raise ValueError(f"File {file} does not exist.")
    print(f"Files to be uploaded: {len(files)}")
    for file in files:
        print(f"  - {file}")
    
    # Create the repository.
    print(f"Creating repository {username_or_orga}/{repo_name}...")
    create_repo(
        f"{username_or_orga}/{repo_name}",
        private=private,
    )

    # Push the files.
    api = HfApi()
    print(f"Pushing files to repository {username_or_orga}/{repo_name}...")
    for file in files:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),
            repo_id=f"{username_or_orga}/{repo_name}",
            repo_type="model",
        )

    # Done.
    print(f"Pushed {len(files)} files to {username_or_orga}/{repo_name}.")


# Entry point.
if __name__ == "__main__":
    fire.Fire(push_to_huggingface)
