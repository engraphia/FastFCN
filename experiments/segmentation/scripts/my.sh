#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset my \
    --model psp --jpu --aux \
    --backbone resnet50 --checkname my \
    --split train --mode train \
    --epochs 1 --lr 0.01 \
    --batch-size 2
