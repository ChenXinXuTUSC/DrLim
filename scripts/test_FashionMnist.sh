#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
WORKSPACE_FOLDER=${SCRIPT_PATH}/..
# env
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1

python ${WORKSPACE_FOLDER}/test.py \
    --data_type FashionMnist \
    --data_root /home/hm/fuguiduo/datasets/fashion-mnist \
    --data_splt test \
    --rslt_root ${WORKSPACE_FOLDER}/images \
    --stat_dict ${WORKSPACE_FOLDER}/runs/stat/2023-05-13_19:32:20_FashionMnist@Contrastive_i1o2e50b8lr0.01/50.pth \
    --loss_type Contrastive \
    --in_channels 1 \
    --out_channels 2 \
    --trans_rsiz 32 \
    --num_workers 4 \
    --batch_size 8 \
