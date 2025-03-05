#!/bin/bash

dataset="mimic_cxr"
annotation="/home/guest/czl/dataset/mimic_cxr/mimic_annotation_all.json"
base_dir="/home/guest/czl/dataset/mimic_cxr/images"
task="report"
# delta_file="/home/guest/czl/R2GenGPT-multistage/save/mimic_cxr/label/label_all_missing_label/checkpoints/checkpoint_epoch0_step1880_bleu1.000000_cider0.000000_ori.pth"

version="report_swin_pmc_llama"
savepath="./save/$dataset/report/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python3 -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --task ${task} \
    --batch_size 24 \
    --val_batch_size 24 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 10 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 0 \
    2>&1 |tee -a ${savepath}/log_mimic.txt
