#!/bin/bash

# ---- Model: GPT 5.4 mini

# ---- Task: Global / [S1, S2, S3] subtask splits

# ---- Settings: [best run ensembling for S1]

# FSL (unified n=4) + ZSL

# 1) Qwen 3.5 4B (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-qwen3.54B-few-global.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

python -m src.grace.eval.score --predictions model/best_runs/ensemble/bestrun_qwen3.54B-few-global-gpt-5.4-mini_few_shot_grace.json --gold data/grace/track_2_dev.json

# 2) MedGemma 4B (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-medgemma4B-few-global.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

# 3) MT5-Base (finetuning split subtask)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-mT5-ft-s1-s2.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

# 3) Gemini-3-Flash (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-gemini_3_flash-few-global.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

# 3) Medgemma 4B (finetuning split subtask)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace_cm-medgemma_4b-ft-s2.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4
