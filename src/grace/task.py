# task.py
# ---------------------------------------------------------------------------------------------
# zero-shot and few-shot prompting and pipeline run for all 3 GRACE subtasks on CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from src.grace.eval import GraceEvaluator
from src.grace.model import get_model, MODEL_FACTORY
from src.case import load_cases, load_relations

import gc
import json
import torch
import random
import logging
import src.config as settings
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def run_subtasks(model_type: str, sizes: list[str], prompt_settings: list[str], tasks: list[str] = ["S1", "S2", "S3"]):
    """Runs the prompting pipeline for all specified subtasks and settings for a given model."""    
    train_cases, train_relations, test_cases, test_relations = _load()
    
    for size in sizes:
        model, model_prefix = get_model(model_type, size, settings.BASE_DATA_DIR)
            
        logging.info(f"\n========================================================")
        logging.info(f"{model_prefix.upper()}-{size} / EVALUATION")
        logging.info(f"========================================================")
        
        for setting in prompt_settings:
            examples = train_cases if setting == "few_shot" else None
            logging.info(f"\n\t >>> [{setting.upper()}] ---")
            
            for task_id in tasks:
                run_func = getattr(model, f"run_subtask_{task_id[-1]}")
                data = test_relations if task_id == "S3" else test_cases
                
                results = run_func(data, few_shot_examples=examples)
                
                out_path = settings.get_prediction_path(model_prefix, size, setting, task_id)
                _save(results, out_path)
            
        logging.info(f"\t> Clearing {model_prefix}-{size}...")
        del model.model
        del model.tokenizer
        del model
        torch.cuda.empty_cache()
        gc.collect()

def evaluate_subtasks(model_type: str, model_size: str, setting: str, tasks: list[str] = ["S1", "S2", "S3"], lang: str = "es"):
    """Evaluates the predictions for all specified subtasks and settings for a given model."""
    evaluator = GraceEvaluator()
    config_entry = MODEL_FACTORY.get(model_type.lower())
    model_prefix = config_entry["prefix"] if config_entry else "Qwen"

    logging.info("\n========================================================")
    logging.info(f"EVALUATING RESULTS: {model_prefix}-{model_size} / {setting}")
    logging.info("========================================================")

    gt_cases = settings.SPLITS_DATA_DIR / "test" / f"test_{lang}_ordered.jsonl"
    gt_relations = settings.SPLITS_DATA_DIR / "test" / f"test_{lang}_relations.jsonl"

    for task_id in tasks:
        pred_path = settings.get_prediction_path(model_prefix, model_size, setting, task_id)
        clean_path = settings.get_prediction_path(model_prefix, model_size, setting, task_id, cleaned=True)
        
        target_path = clean_path if clean_path.exists() else pred_path
        
        if target_path.exists():
            gt_path = gt_relations if task_id == "S3" else gt_cases
            eval_func = getattr(evaluator, f"evaluate_subtask_{task_id[-1]}")
            eval_func(predictions_path=target_path, ground_truth_path=gt_path)
        else:
            logging.warning(f"\t> Missing predictions: {pred_path.name}")

# --- io -------------------------------------------------------------------------

def _save(data: List[Dict[str, Any]], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t >>> Saved results to {out_file.name}")
    
def _load(n: int = 4):
    """Loads cases and relations from the JSON splits."""
    
    train_path = settings.GRACE_SPLITS["train"]
    test_path = settings.GRACE_SPLITS["validation"]
    
    test_cases = load_cases(test_path)    
    train_cases = load_cases(train_path)
    test_relations = load_relations(test_path)
    train_relations = load_relations(train_path)
    
    train_cases.sort(key=lambda x: str(x.get("id", "")))
    train_relations.sort(key=lambda x: str(x.get("id", "")))
    
    random.seed(42)
    fs_cases = random.sample(train_cases, n) if len(train_cases) >= n else train_cases
    fs_relations = random.sample(train_relations, n) if len(train_relations) >= n else train_relations
    
    return fs_cases, fs_relations, test_cases, test_relations