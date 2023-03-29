#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1

python ${SCRIPT_PATH}/../train.py \
    --data_type Cifar10 \
    --data_root /home/hm/fuguiduo/datasets/cifar10 \
    --data_splt train \
    --tfxw_root ${WORKSPACE_FOLDER}/runs/tfxw \
    --stat_root ${WORKSPACE_FOLDER}/runs/stat \
    --num_epochs 100 \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-2 \
    --in_channels 3 \
    --out_channels 3
