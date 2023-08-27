# Title go here

<div align="center">
TODO: Put shields and links here.
</div>

## Abstract from the paper



## Limitations on further development

This is a research project, and so the scope of it is limited to the paper it is attached to. It is unlikely that I will maintaining this repository to ensure it continues to work with the latest dependencies outside those pinned. That said, I've tried to be lax in my pinning of deps so that building on this project will not be overly restrictive.

Additionally, I've tried to work in a constrainted, clean, and robust manner. I hope that helps you as much as it helped me.

## What is included?

Everything. You should be able to run every single experiment from the paper. Datasets and models are hosted on HF. Logs and metrics from training are on WandB so if reproducing, you can check if your model follows mine.

While I tried to bring everything front and centre, some thigns might be buried. If you think these things should be brought forward, feel free to open a PR and bring them forward! I'll definitely be taking opinions regarding this into consideration for future projects.

## Installing this project

Everything you need to know about the installing the dependencies for this project can be found in the `pyproject.toml`. To quickly install and get up and running, you can run the following:

```bash
pyenv install
(add commands for pdm)
```


## How to find out how I did things yourself

Everything that was run, in some shape or form, can be found using the command `python -m cogelot`. This is what was used to run the dataset creation, train models, evaluate models, and more. These commands are implemented using Typer, and can be found within `src/cogelot/entrypoints/__main__.py`.

Separately, I developed everything using tests to verify that each pieces works in isolation and together. You can find all the tests in the `tests/` folder. If you are using an IDE (like VSCode), it likely has support for pytest and the other test-related dependencies I used. While coverage is not going to be 100%, I used the tests with breakpoints to verify inner functions are working as expected and

While there is no separate documentation site, I have tried to make sure that docstrings and comments are relevant and detailed. If you want more information on what a function is doing or why it is doing that, feel free to make an issue. If you figure out something that I haven't described enough off, feel free to make a PR improving my documentation so that you, me, _and_ future people can benefit from your insight.

### Batch files with the commands I ran on SLURM

It is very unlikely that I ran things in a tmux session and just stared at it. I don't like copy-pasting hundreds of commands. As experiments were often run on a compute cluster, I ran commands with SLURM. You can find these contained batch files in `./scripts/slurm/`. These were made for my system, so some adjustments are likely going to be needed, but I'm hoping it's obvious and not too complicated!


## History

### How I created the dataset

The raw data was downloaded from VIMA. Each instance is a folder of multiple files. So that things can be run quickly, the dataset was loaded and parsed with Pydantic, and then converted into a [HF dataset](https://huggingface.co/datasets/amitkparekh/vima). There are unit tests showing how this was done in `tests/test_dataset_creation.py`.

There were some errors when doing it all in one step, so the dataset creation needs to be done in two steps. First, the raw data is parsed and pickled into individual files because this takes the longest to do. Then, these files are loaded and converted into a HF dataset.

The raw dataset is aronud 712GB.

### Tokenising the instances, ready for modelling

To make loading data efficient when modelling, all the instances were tokenized in advanced. Similarly, this is also available on HF, as a different config name.
