#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

pip install -r requirements.txt
# conda install faiss-gpu cudatoolkit=10.2 -c pytorch
# conda install pytorch cudatoolkit=10.2 -c pytorch
conda install faiss-gpu cudatoolkit=11.3 -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

python setup.py develop