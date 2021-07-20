#!/bin/bash

conda create -n pytorch1.8_p36 python=3.6
conda activate pytorch1.8_p36

pip install torch=1.8.1
pip install --disable-pip-version-check -r requirements.txt
pip install --disable-pip-version-check -r triton/requirements.txt --ignore-installed
pip install pytorch-lightning==1.0.0 --no-dependencies
pip install monai==0.4.0 --no-dependencies
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110==0.30.0
pip install torch_optimizer==0.0.1a15 --no-dependencies
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -qq awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install Perf Client required library
sudo apt-get update && apt-get install -y libb64-dev libb64-0d
