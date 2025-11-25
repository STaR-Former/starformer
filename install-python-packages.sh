#!/bin/bash

#python -m pip install --upgrade pip
#
## Check if requirements.txt exists
#if [ -f "requirements-cuda.3.10.14.txt" ]; then
#    echo "requirements-cuda.3.10.14.txt found. Installing dependencies from the file..."
#    pip install -r requirements-cuda.3.10.14.txt
#else
#    echo "requirements-cuda.*.txt not found. Installing directly ..."
#    # pip install cuda versions of torch
#    pip3 install torch torchvision torchaudio
#    pip install -r requirements.3.10.14.txt
#fi

# from source
# 1. build torch form source

# 2. pip install torchvison torchaudio

# reinstall torch from source

# python version 3.13.0
# for pyarrow --> https://arrow.apache.org/docs/developers/python.html#building-on-linux-and-macos
# first

#
# Install all requirements
#

pip install awswrangler aiobotocore boto3 botocore s3fs
pip install pyarrow
pip install hydra-core jupyter jupyterlab \
lightning matplotlib numpy pandas plotly scikit-learn \
tqdm ujson wandb yupi patool pyproj s3fs
pip install -U aeon

# install torchvision
#pip install torch torchvision torchaudio



