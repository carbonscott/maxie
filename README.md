<div align="center">

# MAXIE - Masked X-Ray Image AutoEncoder

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<!--<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> -->
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->

</div>

## Description

MAXIE contains machinery to build, train, and run
AI that operates on masked X-ray images.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/carbonscott/maxie
cd maxie

# [OPTION 1] create python virtual environment
python3 -m venv ./myenv
. ./myenv/bin/activate
export PIP_CACHE_DIR=/tmp/$USER

# [OPTION 2] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install mpi4py and h5py
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
HDF5_MPI="ON" CC=cc HDF5_DIR=${HDF5_ROOT} pip install --no-cache-dir --no-binary=h5py h5py

# install requirements
pip install -r requirements.txt

# install maxie (-e for editable install)
pip install -e .
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
