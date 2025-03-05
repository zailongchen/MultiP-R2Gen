#!/bin/bash

dataset="mimic_cxr"
annotation="/home/guest/czl/dataset/mimic_cxr/mimic_annotation_all.json"
base_dir="/home/guest/czl/dataset/mimic_cxr/images"
task="report"
delta_file="/home/guest/czl/R2GenGPT-main/save/mimic_cxr/v1_deep/checkpoints/checkpoint_epoch5_step15510_bleu0.143398_cider0.293064.pth"

version="v1_deep"
savepath="./save/$dataset/$version"

python3 -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 40 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 8 \
    --devices 4 \
    --limit_test_batches 1.0 \
    2>&1 |tee -a ${savepath}/log_mimic_test.txt
