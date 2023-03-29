#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1

python ${SCRIPT_PATH}/../test.py \
    --data_type Cifar10 \
    --data_root /home/hm/fuguiduo/datasets/cifar10 \
    --data_splt test \
    --rslt_root ${WORKSPACE_FOLDER}/results \
    --stat_dict ${WORKSPACE_FOLDER}/runs/stat/2023-03-29_21:10:59_Cifar10_i3o3e100b8lr0.01/80.pth \
    --num_workers 4 \
    --batch_size 8 \
    --in_channels 3 \
    --out_channels 3
