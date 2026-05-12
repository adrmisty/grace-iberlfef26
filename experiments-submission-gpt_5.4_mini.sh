#!/bin/bash

# ---- Model: GPT 5.4 mini

# ---- Task: Inference over blind test for submission (task S3)

# 1) Qwen 3.5 4B (ICL global) // Zero-shot ICL with GRACE
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-qwen3.54B-few-global.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings zero_shot \
  --dataset blind_grace \
  --n_examples 0

# 2) Ensemble top 5 s2 (weighted cluster vote + IoU restriction) // Zero-shot ICL with GRACE
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/ens_s2_top5_weighted_iou7.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings zero_shot \
  --dataset blind_grace \
  --n_examples 0

