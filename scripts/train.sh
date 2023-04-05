#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1
# activate the virtual env before executing this script
python ${SCRIPT_PATH}/../train.py \
    --data_type Cifar10 \
    --data_root /home/hm/fuguiduo/datasets/cifar10 \
    --data_splt train \
    --tfxw_root ${WORKSPACE_FOLDER}/runs/tfxw \
    --stat_root ${WORKSPACE_FOLDER}/runs/stat \
    --loss_type Triplet \
    --in_channels 1 \
    --out_channels 3 \
    --trans_rsiz 32 \
    --trans_gray\
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-2 \
    --num_epochs 50 \
    --save_freq 5 \
    --info_freq 100
