<div align="center">

# Captcha Resolver

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?labelColor=gray)](https://www.python.org/)
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://github.com/Mai0313/text-resolver/actions/workflows/test.yml/badge.svg)](https://github.com/Mai0313/text-resolver/actions/workflows/test.yml)
[![code-quality](https://github.com/Mai0313/text-resolver/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Mai0313/text-resolver/actions/workflows/code-quality-main.yaml)

</div>

## Description

This is a simple template for you to train your own captcha resolver with hydra template.

You can simply change the configuration in [configs/experiment/](configs/experiment/) and change the model, dataset, and trainer in [src/](src/).

For this project, I used a simple CNN model with 3 convolutional layers and 2 fully connected layers, and it achieved 93.57% accuracy on the test set easily.

## Results

### Accuracy: 93.57%

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.lock
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
