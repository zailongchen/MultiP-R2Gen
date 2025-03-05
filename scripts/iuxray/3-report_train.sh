#!/bin/bash

dataset="iu_xray"
annotation="/home/guest/czl/dataset/iu_xray/annotation_report.json"
base_dir="/home/guest/czl/dataset/iu_xray/images"
task="report"
delta_file="/home/guest/czl/R2GenGPT-multistage/save/iu_xray/triple/triple_swin_pmc_llama/checkpoints/checkpoint_epoch14_step1290_bleu0.222505_cider0.688717_ori.pth"

version="report_with_label_triple_pretrain_swin_pmc_llama_f16"
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
    --delta_file ${delta_file} \
    --batch_size 24 \
    --val_batch_size 24 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 15 \
    --limit_train_batches 1.0 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    2>&1 |tee -a ${savepath}/log_mimic.txt
