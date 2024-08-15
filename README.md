# Helibrunna
## A Hugging Face compatible xLSTM trainer by Dr. Tristan Behrens

![](assets/helibrunna.jpg)


## Acknowledgements

This repository is dedicated to my second hometown Heilbronn, who has become one of the most creative AI hubs in Germany.
This work is sponsored by [KI Salon](https://www.ki-salon.net/), who is a strong supporter of open-source AI.
We have built the functionality on top of the official [xLSTM](https://github.com/NX-AI/xlstm) project.

## Get in touch

Do not hesitate to report any issues that you might find [here](https://github.com/AI-Guru/helibrunna/issues). Of course, do not hesitate to connect on [LinkedIn](https://de.linkedin.com/in/dr-tristan-behrens-734967a2) and say hello.

## Features

Note, that as of now, this implementation is quite basic. It is our goal to accelerate the adoption of xLSTM to find out how superior it is to self-attention based transformers (if it is). This goal requires thorough experimentation.

In other words: This repo is currently in an early stage, and thus we cannot guarantee that it works.

These features are currently implemented:

- Training xLSTM with datasets hosted on [Hugging Face](https://huggingface.co/).
- Support for [TensorBoard](https://www.tensorflow.org/tensorboard).
- Support for [Weights & Biases](https://wandb.ai/home).
- Distributed training [Accelerate](https://huggingface.co/docs/accelerate/index) (untested).

These features are planned:

- Running model inference.
- Exposing the model as an OpenAI compatible API.
- Fine-tuning.
- Fine-tuning with LoRA adapters.
- Quantization.
- Uploading to Hugging Face.
- Downloading from Hugging Face.
- Training on a GPT2-size dataset, such as openwebtext.


## Known issues


## Setting up things

First, be so kind and install xLSTM following the instructions here: https://github.com/NX-AI/xlstm

This should be a walk in the park. Do not skip the step with the conda environment and please make sure this environment is active.

Then, please install additional dependencies using `requirements.txt`:

```
conda activate xlstm
pip install -r requirements.txt
```

Then you should be ready to go!


## Training xLSTM

Here, we will collect a few examples. Make sure that the conda environment is active.


### Training a music-xLSTM on Johann Sebastian Bach chorales

```
python train.py configs/musicxlstm.yaml
```

## Running inference