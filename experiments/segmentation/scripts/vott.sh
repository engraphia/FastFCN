#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset vott \
    --model fcn --jpu --aux \
    --checkname vott \
    --split train --mode train \
    --epochs 30 --lr 0.01 \
    --batch-size 5
