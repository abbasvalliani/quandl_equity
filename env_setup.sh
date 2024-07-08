#!/bin/sh

conda env remove --name ml_env
conda create --name ml_env python=3.12
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install pip
pip install -r requirements.txt
