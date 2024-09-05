# Helibrunna
## A Hugging Face compatible xLSTM trainer by Dr. Tristan Behrens

![](assets/helibrunna02.jpg)

This is how ChatGPT describes the project:

> Helibrunna is an advanced, open-source framework designed to facilitate the training and experimentation of xLSTM models. Developed by Dr. Tristan Behrens, this tool aims to explore the potential superiority of xLSTM architectures over traditional self-attention-based transformers. It is especially tailored for use with datasets hosted on Hugging Face, making it a versatile tool for developers and researchers in the AI community.

Nice!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AI-Guru/helibrunna&type=Date)](https://star-history.com/#AI-Guru/helibrunna&Date)


## License Overview

This project is licensed under the GNU Affero General Public License (AGPL) version 3. We have chosen this license to maintain consistency with the xLSTM project, which is also licensed under the AGPL. 

The AGPL is specifically designed to ensure that any modifications to the code, especially when deployed in a networked environment, are shared with the community. This aligns with the principles of the xLSTM project, promoting open collaboration and ensuring that improvements remain freely accessible to all users.

For more details on the license, please see the [LICENSE](./LICENSE) file.


## Acknowledgements

This repository is dedicated to my second hometown [Heilbronn](https://www.heilbronn.de/startseite.html), who has become one of the most creative AI hubs in Germany.

This work is sponsored by [KI Salon](https://www.ki-salon.net/), who is a strong supporter of open-source AI.

We have built the functionality on top of the official [xLSTM](https://github.com/NX-AI/xlstm) project.

Shoutout to [experimenta](https://www.experimenta.science/), [Bildungscampus](https://www.bildungscampus.hn/), [42 coding school](https://www.42heilbronn.de/), [IPAI](https://ip.ai/), [STACKIT](https://www.stackit.de/de/) and [Dieter Schwarz Stiftung](https://www.dieter-schwarz-stiftung.de/), who among others make Heilbronn a high-tech place.


## Get in touch and get involved

Do not hesitate to report any issues that you might find [here](https://github.com/AI-Guru/helibrunna/issues). And please connect on [LinkedIn](https://de.linkedin.com/in/dr-tristan-behrens-734967a2). We are happy about everyone who says "hello".

If you want to contribute, please fork the repository and send pull requests. Looking forward! Since this is an open-source project we are most eager for you to participate.

And if you want to join as a developer, let us know!


## How to use, how to credit, and README

We would be very happy if you would go wild and train xLSTMs with Helibrunna. If you publish your work, please be so kind and give credit to Helibrunna and link the project.

You can use this banner:

![Trained with Helibrunna](assets/trainedwithhelibrunna.jpg)

And this is an example of how to credit:

> Trained with [Helibrunna](https://github.com/AI-Guru/helibrunna)

Note, that everytime you train, a template README (aka modelcard) file will be generated. You can and you should edit it before uploading your models anywhere. Here is an example: [musicxlstm on Hugging Face](https://huggingface.co/TristanBehrens/musicxlstm).

And please, if you have published anything, let us know. We would love to promote your work.


## Features

Note, that as of now, this implementation is quite basic. It is our goal to accelerate the adoption of xLSTM to find out how superior it is to self-attention based transformers (if it is). This goal requires thorough experimentation.

In other words: This repo is currently in an early stage, and thus we cannot guarantee that it works.

These features are currently implemented:

- Training xLSTM with datasets hosted on [Hugging Face](https://huggingface.co/).
- Support for [Weights & Biases](https://wandb.ai/home).
- Distributed training [Accelerate](https://huggingface.co/docs/accelerate/index) (untested).
- Basic model inference with temperature sampling.
- Uploading to Hugging Face.
- Downloading from Hugging Face.

These features are planned or would be great to work on:

- Exposing the model as an OpenAI compatible API.
- Support for [TensorBoard](https://www.tensorflow.org/tensorboard).
- Fine-tuning.
- Fine-tuning with LoRA adapters.
- Quantization.
- Training on a GPT2-size dataset, such as openwebtext.
- More sophisticated sampling.
- Porting to MLX.


## Setting up things

So far, when it comes to compatibility, we have these configurations:

|                | Apple (no silicon) | Apple (silicon) | Unix (NVIDIA) | Unix (no NVIDIA) | Raspberry Pi    | Windows |
|----------------|--------------------|-----------------|---------------|------------------|-----------------|---------|
| xLSTM          | ❔                 | ❌              | ✅             | ❔               | ❌               | ❔      |
| Mamba          | ❔                 | ❌              | ✅             | ❔               | ❌               | ❔      |
| Pharia         | ❔                 | ✅              | ✅             | ❔               | ✅               | ❔      |
| Transformer    | ❔                 | ✅              | ✅             | ❔               | ✅               | ❔      |

✅ = tested and working, ❌ = tested and not working, ❔ = not tested

### Unix (NVIDIA)

First, be so kind and install xLSTM following the instructions here: https://github.com/NX-AI/xlstm

This should be a walk in the park. Do not skip the step with the conda environment and please make sure this environment is active.

Then, please install additional dependencies using `requirements.txt`:

```
conda activate xlstm
pip install -r requirements.txt
```

Then you should be ready to go!

### Apple (silicon)

Support is not fully implemented yet. We are working on it.

It is advised to create a new conda environment and install the dependencies from `requirements.txt`:

```
conda create -n "helibrunna" python=3.10.13
conda activate helibrunna
pip install -r requirements-mac.txt
```

### Raspberry Pi

Support is not fully implemented yet. We are working on it.

It is advised to create a new conda environment and install the dependencies from `requirements.txt`:

```
conda create -n "helibrunna" python=3.12.4
conda activate helibrunna
pip install -r requirements-raspberry.txt
```

## Dataset Preprocessing

Usually the dateset preprocessing happens very early when you start a training. Some datasets might require you to preprocess in a separate step. This is how you can do it:

```
python train.py preprocess configs/musicxlstm.yaml
```

## Training xLSTM

Here, we will collect a few examples. Make sure that the conda environment is active.


### Training a music-xLSTM on Johann Sebastian Bach chorales

We have included a config file that will train xLSTM on symbolic music. You can run it like this:

```
python train.py configs/musicxlstm.yaml
```


### Training an xLSTM on the written works by H.P. Lovecraft

There is also another config file that upcycles the GPT2 tokenizer and trains an xLSTM on the [lovecraft corpus](https://huggingface.co/datasets/TristanBehrens/lovecraftcorpus):


```
python train.py configs/lovecraft.yaml
```

## Training with Accelerate

```
accelerate config
accelerate launch train.py configs/musicxlstm.yaml
```

## Train on openwebtext

This one is a little more complex. Because of the dataset size it is advised to preprocess it first. And here the model config is standalone and must be loaded separatedly.

This is preprocessing with the separate model config:

```
python train.py preprocess configs/openwebtext.yaml configs/xlstm_7to1_01_125M.yaml
```

And this is how to train:

```
accelerate launch train.py configs/openwebtext.yaml configs/xlstm_7to1_01_125M.yaml
```

## Running inference

This is how you can run inference with a trained model:

```
python generate.py --model_path_or_repo MODEL_PATH_OR_REPO --temperature 0.5 --max_length 100 --prompt "PROMPT"
```

Set `MODEL_PATH_OR_REPO`, and `PROMPT` properly. `MODEL_PATH_OR_REPO` is usually a directory that starts with `run_`, or of course and xLSTM that lives on Hugging Face.

Here is an example that will download and run [musicxlstm](https://huggingface.co/TristanBehrens/musicxlstm).

```
python generate.py --model_path_or_repo TristanBehrens/musicxlstm --temperature 0.5 --max_length 100 --prompt "PIECE_START"
```

## Uploading a model to Hugging Face.

Make sure that you are logged into Hugging Face. If you are not, do this:

```
huggingface-cli login
```

Make sure you use an access token that allows for writing.

This is how you can push a model. It will use the latest checkpoint:

```
python pushtohuggingface.py --model_path MODEL_PATH --username_or_orga USERNAME_OR_ORGA --repo_name REPO_NAME --private true
```

Make sure to fill in `MODEL_PATH`, `USERNAME_OR_ORGA`, and `REPO_NAME`. `MODEL_PATH` is usually a directory that starts with `run_`.

You might want to edit the `README.md` file.

## Inference speeds

Here are some inference speeds for the models that we have trained. Unit of measurement is tokens per second:

|                | Apple (no silicon) | Apple (silicon) | Unix (NVIDIA) | Unix (no NVIDIA) | Raspberry Pi    | Windows |
|----------------|--------------------|-----------------|---------------|------------------|-----------------|---------|
| xLSTM          | ❔                 | ❔              | 23             | ❔               | ❔              | ❔       |
| Mamba          | ❔                 | ❔              | 237            | ❔               | ❔              | ❔       |
| Pharia         | ❔                 | 688             | 364            | ❔               | 51              | ❔       |
| Transformer    | ❔                 | 980             | 528            | ❔               | 64              | ❔       |

These are the models that we have tested:

- https://huggingface.co/TristanBehrens/bach-garland-xlstm
- https://huggingface.co/TristanBehrens/bach-garland-mamba
- https://huggingface.co/TristanBehrens/bach-garland-pharia
- https://huggingface.co/TristanBehrens/bach-garland-transformer


Apple (Silicon):

- Platform: Darwin
- Platform Version: Darwin Kernel Version 23.3.0: Wed Dec 20 21:31:10 PST 2023; root:xnu-10002.81.5~7/RELEASE_ARM64_T6031
- Architecture: arm64
- Processor: arm
- Python Version: 3.10.13
- CPU Cores (Logical): 16
- CPU Cores (Physical): 16
- Total Memory (GB): 48.0
- Total Disk Space (GB): 926.3517189025879
- GPU: No GPU detected

Unix (NVIDIA):

- Platform: Linux
- Platform Version: #129~20.04.1-Ubuntu SMP Wed Aug 7 13:07:13 UTC 2024
- Architecture: x86_64
- Processor: x86_64
- Python Version: 3.10.9
- CPU Cores (Logical): 24
- CPU Cores (Physical): 12
- Total Memory (GB): 31.250316619873047
- Total Disk Space (GB): 915.3232879638672
- GPU 0 Name: NVIDIA GeForce RTX 3090
- GPU 0 Memory Total (GB): 24.0
- GPU 0 Driver Version: 535.161.08
- GPU 1 Name: NVIDIA GeForce RTX 3090
- GPU 1 Memory Total (GB): 24.0
- GPU 1 Driver Version: 535.161.08

Raspberry Pi:

- Platform: Linux
- Platform Version: #1 SMP PREEMPT Debian 1:6.6.31-1+rpt1 (2024-05-29)
- Architecture: aarch64
- Processor: 
- Python Version: 3.12.4
- CPU Cores (Logical): 4
- CPU Cores (Physical): 4
- Total Memory (GB): 7.8636627197265625
- Total Disk Space (GB): 116.66678619384766
- GPU: GPUtil not installed or no GPU detected


# THANKS!
