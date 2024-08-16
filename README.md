# Helibrunna
## A Hugging Face compatible xLSTM trainer by Dr. Tristan Behrens

![](assets/helibrunna02.jpg)

This is how ChatGPT describes the project:

> Helibrunna is an advanced, open-source framework designed to facilitate the training and experimentation of xLSTM models. Developed by Dr. Tristan Behrens, this tool aims to explore the potential superiority of xLSTM architectures over traditional self-attention-based transformers. It is especially tailored for use with datasets hosted on Hugging Face, making it a versatile tool for developers and researchers in the AI community.

Nice!

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
- Support for [TensorBoard](https://www.tensorflow.org/tensorboard).
- Support for [Weights & Biases](https://wandb.ai/home).
- Distributed training [Accelerate](https://huggingface.co/docs/accelerate/index) (untested).
- Basic model inference with temperature sampling.
- Uploading to Hugging Face.
- Downloading from Hugging Face.

These features are planned or would be great to work on:

- Exposing the model as an OpenAI compatible API.
- Fine-tuning.
- Fine-tuning with LoRA adapters.
- Quantization.
- Training on a GPT2-size dataset, such as openwebtext.
- More sophisticated sampling.
- Porting to MLX.


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

We have included a config file that will train xLSTM on symbolic music. You can run it like this:

```
python train.py configs/musicxlstm.yaml
```


### Training an xLSTM on the written works by H.P. Lovecraft

There is also another config file that upcycles the GPT2 tokenizer and trains an xLSTM on the [lovecraft corpus](https://huggingface.co/datasets/TristanBehrens/lovecraftcorpus):


```
python train.py configs/lovecraft.yaml
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

# THANKS!