#!/bin/bash

# ---- Model: MedGemma 4B

# ---- Task: Global

# ---- Settings 

# ZSL
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  zero_shot --tasks global &> medgemma4B_global_zsl.log


# FSL - grace n=4
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset grace --n_examples 4 &> medgemma4B_global_fsl4_grace.log

# FSL - balanced n=4
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset unified --n_examples 4 &> medgemma4B_global_fsl4_unified.log

# FSL - balanced n=4
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset casimedicos --n_examples 4 &> medgemma4B_global_fsl4_casimedicos.log

# FSL - grace n=8
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset grace --n_examples 8 &> medgemma4B_global_fsl8_grace.log

# FSL - balanced n=8
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset unified --n_examples 8 &> medgemma4B_global_fsl8_unified.log

# FSL - balanced n=8
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID uv run -m src.grace.main --run --post --submit \
    --model MedGemma --sizes 4B --settings  few_shot --tasks global --dataset casimedicos --n_examples 8 &> medgemma4B_global_fsl8_casimedicos.log
