#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1

python ${WORKSPACE_FOLDER}/test.py \
    --data_type Cifar10 \
    --data_root /home/hm/fuguiduo/datasets/cifar10 \
    --data_splt test \
    --rslt_root ${WORKSPACE_FOLDER}/results \
    --stat_dict ${WORKSPACE_FOLDER}/runs/stat/2023-04-05_14:41:53_Cifar10@Triplet_i1o3e50b8lr0.01/30.pth \
    --loss_type Triplet \
    --in_channels 1 \
    --out_channels 3 \
    --trans_rsiz 32 \
    --trans_gray \
    --num_workers 4 \
    --batch_size 8 \
