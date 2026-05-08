#!/bin/bash

# ---- Model: GPT 5.4 mini

# ---- Task: Global / [S1, S2, S3] subtask splits

# ---- Settings 

# ZSL
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python -m src.grace.main --run --post --submit \
    --model OpenAI --sizes gpt-5.4-mini --settings  zero_shot --tasks global S1 S2 S3 &> gpt5.4mini_global_zsl.log

# FSL - grace n=4
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model OpenAI --sizes gpt-5.4-mini --settings  few_shot --tasks global S1 S2 S3 --dataset grace --n_examples 4 &> gpt5.4mini_global_fsl4_grace.log

# FSL - balanced n=4
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model OpenAI --sizes gpt-5.4-mini --settings  few_shot --tasks global S1 S2 S3 --dataset unified --n_examples 4 &> gpt5.4mini_global_fsl4_unified.log