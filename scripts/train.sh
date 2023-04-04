#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1
# activate the virtual env before executing this script
python ${SCRIPT_PATH}/../train.py \
    --data_type FashionMnist \
    --data_root /home/hm/fuguiduo/datasets/fashion-mnist \
    --data_splt train \
    --tfxw_root ${WORKSPACE_FOLDER}/runs/tfxw \
    --stat_root ${WORKSPACE_FOLDER}/runs/stat \
    --in_channels 1 \
    --out_channels 2 \
    --trans_rsiz 32 \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-2 \
    --num_epochs 10 \
    --save_freq 1 \
    --info_freq 100
