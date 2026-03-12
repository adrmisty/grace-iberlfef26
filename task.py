# task.py
# ----------------------------------------------------------------------------------------
# zero-shot and few-shot prompting and pipeline run on CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from eval import GraceEvaluator
from model import GraceModel
from case import load_cases, load_relations

import gc
import json
import torch
import random
import logging
import config as settings
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")


def run_subtasks(sizes: list[str], prompt_settings: list[str]):    
    # ** FEW-SHOT EXAMPLES AND MODEL SET UP **
    train_cases, train_relations, test_cases, test_relations = _load(n=3)
    
    for size in sizes:
        logging.info(f"\n========================================================")
        logging.info(f"QWEN3.5-{size} / FEW-SHOT / ZERO-SHOT EVALUATION")
        logging.info(f"========================================================")
        
        grace = GraceModel(model_size=size)
        
        for setting in prompt_settings:
            if setting == "few_shot":
                examples = train_cases
                
            logging.info(f"\n\t >>> [{setting.upper()}] ---")
            
            # --- [1] ---
            s1_results = grace.run_subtask_1(test_cases, few_shot_examples=examples)
            _save(s1_results, settings.MODEL_DIR / size / f"Qwen_{size}_{setting}_S1.json")
            
            # --- [2] ---
            s2_results = grace.run_subtask_2(test_cases, few_shot_examples=examples)
            _save(s2_results, settings.MODEL_DIR / size / f"Qwen_{size}_{setting}_S2.json")
            
            # --- [3] ---
            s3_results = grace.run_subtask_3(test_relations, few_shot_examples=examples)
            _save(s3_results, settings.MODEL_DIR / size / f"Qwen_{size}_{setting}_S3.json")
            
        # ** clear memory upon model size change **
        logging.info(f"\t> Clearing Qwen3.5-{size}...")
        del grace.model
        del grace.tokenizer
        del grace
        torch.cuda.empty_cache()
        gc.collect()

def evaluate_subtasks(model_size: str, setting: str, MODEL_DIR: Path = settings.MODEL_DIR, splits_dir: Path = settings.SPLITS_DATA_DIR):
    """
    Evaluates all three subtasks for a given experiment run."""

    evaluator = GraceEvaluator()

    logging.info("\n========================================================")
    logging.info(f"EVALUATING RESULTS: Qwen-{model_size} / {setting}")
    logging.info("========================================================")

    # prediction files from run_subtasks
    s1_path = MODEL_DIR / model_size / f"Qwen_{model_size}_{setting}_S1.json"
    s2_path = MODEL_DIR / model_size / f"Qwen_{model_size}_{setting}_S2.json"
    s3_path = MODEL_DIR / model_size / f"Qwen_{model_size}_{setting}_S3.json"

    # ground truth files
    gt_cases = splits_dir / "test" / "test_es_ordered.jsonl"
    gt_relations = splits_dir / "test" / "test_es_relations.jsonl"

    # --- [1] ---
    if s1_path.exists():
        evaluator.evaluate_subtask_1(
            predictions_path=s1_path,
            ground_truth_path=gt_cases
        )
    else:
        logging.warning(f"\t> Missing predictions: {s1_path}")

    # --- [2] ---
    if s2_path.exists():
        evaluator.evaluate_subtask_2(
            predictions_path=s2_path,
            ground_truth_path=gt_cases
        )
    else:
        logging.warning(f"\t> Missing predictions: {s2_path}")

    # --- [3] ---
    if s3_path.exists():
        evaluator.evaluate_subtask_3(
            predictions_path=s3_path,
            ground_truth_path=gt_relations
        )
    else:
        logging.warning(f"\t> Missing predictions: {s3_path}")

# --- io -------------------------------------------------------------------------

def _save(data: List[Dict[str, Any]], out_file: Path):
    """Saves evaluation predictions to disk."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t >>> Saved results to {out_file.name}")
    

def _load(n: int = 3):
    """Loads few-shot examples for prompting after parsing clinical cases and relations."""
    # ** SUBTASKS 1 AND 2: CASE PARSING **
    test_dir = settings.SPLITS_DATA_DIR / "test"
    train_dir = settings.SPLITS_DATA_DIR / "train"

    test_cases = load_cases(test_dir / "test_es_ordered.jsonl")    
    train_cases = load_cases(train_dir / "train_es_ordered.jsonl")
    
    # ** SUBTASK 3: RELATION LOADING **
    test_relations = load_relations(test_dir / "test_es_relations.jsonl")
    train_relations = load_relations(train_dir / "train_es_relations.jsonl")
    
    random.seed(42)
    train_cases = random.sample(train_cases, n) if len(train_cases) >= n else train_cases
    train_relations = random.sample(train_relations, n) if len(train_relations) >= n else train_relations
        
    return train_cases, train_relations, test_cases, test_relations