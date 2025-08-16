#!/bin/bash
source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate alignprop

# HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo.py --config configs/spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10.py
# HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_drtune.py --config configs/drtune_sd-v1-4.py
HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo.py --config configs/spo_sd-v1-4.py