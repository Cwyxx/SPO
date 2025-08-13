#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

aigi_detector="dinov2"
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 train_scripts.py \
    --tracker_project_name "${aigi_detector}_training" \
    --tracker_experiment_name "genimage" \
    --aigi_detector "${aigi_detector}" \
    --learning_rate 1e-05 \
    --output_dir "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/${aigi_detector}/genimage" \
    --dataset_name "genimage" \
    --max_grad_norm 1 \
    --resume_from "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2/genimage/best_model" \
    --first_epoch 1 \
    --epochs 5