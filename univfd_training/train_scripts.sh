#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 train_scripts.py \
    --tracker_project_name "univfd_training" \
    --tracker_experiment_name "genimage" \
    --learning_rate 1e-05 \
    --output_dir "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/univfd/genimage" \
    --dataset_name "genimage" \
    --max_grad_norm 1 \
    --epochs 10