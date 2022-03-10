#!/usr/bin/env bash

conda create -n bert_hw python=3.7
conda activate bert_hw

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit==10.1 -c pytorch
python -m pip install tqdm==4.58.0 requests==2.25.1 importlib-metadata==3.7.0 filelock==3.0.12 sklearn==0.0 tokenizers==0.10.1
