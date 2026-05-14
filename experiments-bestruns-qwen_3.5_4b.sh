#!/bin/bash

# ---- Model: Qwen 3.5 4B

# ---- Task: Global / [S1, S2, S3] subtask splits

# ---- Settings: [best run ensembling for S3]

# 1) Qwen 3.5 4B (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-qwen3.54B-few-global.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset grace \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/qwen3.5-4b/bestrun_qwen3.54B-few-global_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> qwen3.54B-few-global_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/qwen3.5-4b/bestrun_qwen3.54B-few-global_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> qwen3.54B-few-global_zero_shot.log

# 2) MedGemma 4B (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-medgemma4B-few-global.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset grace \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/medgemma-4b-few-global/bestrun_medgemma4B-few-global_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> medgemma4B-few-global_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/medgemma-4b-few-global/bestrun_medgemma4B-few-global_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> medgemma4B-few-global_zero_shot.log


# 3) MT5-Base (finetuning split subtask)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-mT5-ft-s1-s2.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset grace \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/mt5-base-ft/bestrun_mT5-ft-s1-s2_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> mT5-ft-s1-s2_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/mt5-base-ft/bestrun_mT5-ft-s1-s2_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> mT5-ft-s1-s2_zero_shot.log


# 4) Gemini-3-Flash (ICL global)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace-gemini_3_flash-few-global.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/gemini-3-flash/bestrun_gemini_3_flash-few-global_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> gemini_3_flash-few-global_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/gemini-3-flash/bestrun_gemini_3_flash-few-global_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> gemini_3_flash-few-global_zero_shot.log


# 5) Medgemma 4B (finetuning split subtask)
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/grace_cm-medgemma_4b-ft-s2.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/medgemma-4b-ft/bestrun_medgemma_4b-ft-s2_Qwen-4B_few_shot_grace_cm.json --gold data/grace/track_2_dev.json &> medgemma_4b-ft-s2_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/medgemma-4b-ft/bestrun_medgemma_4b-ft-s2_Qwen-4B_zero_shot_grace_cm.json --gold data/grace/track_2_dev.json &> medgemma_4b-ft-s2_zero_shot.log

# ----------------------------------------------- ensembles

# 1) Ensemble top 5 no commercial weighted iou7
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/ens_s2_top5_nocommercial_weighted_iou7.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset grace \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/ens_s2_top5_nocommercial_weighted_iou7/bestrun_ens_s2_top5_nocommercial_weighted_iou7_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> ens_top5_nocommercial_weighted_iou7_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/ens_s2_top5_nocommercial_weighted_iou7/bestrun_ens_s2_top5_nocommercial_weighted_iou7_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> ens_top5_nocommercial_weighted_iou7_zero_shot.log

# 2) Ensemble top 5 weighted iou7
CUDA_VISIBLE_DEVICES=3 python -m src.grace.main \
  --bestrun \
  --other_predictions model/best_runs/ens_s2_top5_weighted_iou7.json \
  --model Qwen \
  --sizes 4B \
  --settings few_shot zero_shot \
  --dataset unified \
  --n_examples 4

python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/ens_s2_top5_weighted_iou7/bestrun_ens_s2_top5_weighted_iou7_Qwen-4B_few_shot_grace.json --gold data/grace/track_2_dev.json &> ens_top5_weighted_iou7_few_shot.log
python -m src.grace.eval.score2 --task 3 --predictions model/best_runs/ensemble_qwen3.5/ens_s2_top5_weighted_iou7/bestrun_ens_s2_top5_weighted_iou7_Qwen-4B_zero_shot_grace.json --gold data/grace/track_2_dev.json &> ens_top5_weighted_iou7_zero_shot.log