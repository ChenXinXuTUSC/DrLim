#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1

python ${SCRIPT_PATH}/../test.py \
    --data_type FashionMnist \
    --data_root /home/hm/fuguiduo/datasets/fashion-mnist \
    --data_splt test \
    --rslt_root ${WORKSPACE_FOLDER}/results \
    --stat_dict ${WORKSPACE_FOLDER}/runs/stat/2023-04-04_20:45:47_FashionMnist_i1o2e10b8lr0.01/3.pth \
    --trans_rsiz 32 \
    --num_workers 4 \
    --batch_size 8 \
    --in_channels 1 \
    --out_channels 2
