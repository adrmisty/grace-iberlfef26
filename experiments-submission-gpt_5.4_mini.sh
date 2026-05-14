#!/bin/bash

# ---- Model: GPT 5.4 mini

# ---- Task: Inference over blind test for submission (task S3)

# 1) Qwen 3.5 4B (ICL global) // Zero-shot ICL with GRACE
# on Paula's results for S2
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-qwen3.54B-few-global_paula.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings zero_shot \
  --dataset blind_grace \
  --n_examples 0

# 2) Ensemble top 5 s2 (weighted cluster vote + IoU restriction) // Zero-shot ICL with GRACE
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/ens_s2_top5_weighted_iou7_alvaro.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot \
  --dataset blind_grace \
  --n_examples 4

# ---- Task: Inference over blind test for submission (task S1)

# 1) GPT-5.4 mini | Few-shot | GRACE + CM (Ejemplos Few Shot) | Score (en eval): 88,89
# grace_cm-gpt_5.4_mini-few-s1
CUDA_VISIBLE_DEVICES=5 python -m src.grace.main \
  --run --tasks S1 --clean --submit \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot \
  --dataset blind_grace \
  --n_examples 4

# 2) GPT-5.4 mini | Zero-shot | GRACE | Score (en eval): 88,52 | Responsable: Adriana
# grace-gpt_5.4_mini-zero-s1
CUDA_VISIBLE_DEVICES=5 python -m src.grace.main \
  --run --tasks S1 --clean --submit \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings zero_shot \
  --dataset blind_grace \
  --n_examples 0

