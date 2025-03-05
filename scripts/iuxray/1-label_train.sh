#!/bin/bash

dataset="iu_xray"
annotation="/home/guest/czl/dataset/iu_xray/annotation_label.json"
base_dir="/home/guest/czl/dataset/iu_xray/images"
task="label"

version="label_swin_pmc_llama_f16"
savepath="./save/$dataset/label/$version"

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
    --batch_size 12 \
    --val_batch_size 12 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 114 \
    --min_new_tokens 110 \
    --max_new_tokens 114 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 2 \
    --max_epochs 15 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 1 \
    2>&1 |tee -a ${savepath}/log_mimic.txt
