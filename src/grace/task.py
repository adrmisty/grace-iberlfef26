# task.py
# ---------------------------------------------------------------------------------------------
# zero-shot and few-shot prompting and pipeline run for all 3 GRACE subtasks on CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from src.grace.eval import GraceEvaluator
from src.grace.model import get_model, MODEL_FACTORY
from src.case import *

import gc
import json
import torch
import random
import logging
import src.config as settings
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def run_subtasks(model_type: str, sizes: list[str], prompt_settings: list[str], tasks: list[str] = ["S1", "S2", "S3"], dataset: str = "grace"):
    """Runs the prompting pipeline for all specified subtasks and settings for a given model."""    
    train_cases, train_relations, test_cases, test_relations = _load(dataset=dataset)
    
    for size in sizes:
        config_entry = MODEL_FACTORY.get(model_type.lower())
        model_prefix = config_entry["prefix"] if config_entry else model_type
        model = MODEL_FACTORY[model_type.lower()]["class"](size)
            
        logging.info(f"\n========================================================")
        logging.info(f"{model_prefix.upper()}-{size} / EVALUATION ({dataset.upper()})")
        logging.info(f"========================================================")
        
        for setting in prompt_settings:
            logging.info(f"\n\t >>> [{setting.upper()}] ---")
            
            for task_id in tasks:
                run_func = getattr(model, f"run_subtask_{task_id[-1]}")
                
                if task_id == "S3":
                    data = test_relations
                    examples = train_relations if setting == "few_shot" else None
                else:
                    data = test_cases
                    examples = train_cases if setting == "few_shot" else None

                lang_code = "en" if dataset == "casimedicos" else "es"
                results = run_func(data, few_shot_examples=examples, lang=lang_code)
                
                out_path = settings.get_prediction_path(model_prefix, size, setting, task_id)
                out_path = out_path.with_name(out_path.name.replace(".json", f"_{dataset}.json")) 
                
                _save(results, out_path)
            
        logging.info(f"\t> Clearing {model_prefix}-{size}...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
                
def evaluate_subtasks(model_type: str, model_size: str, setting: str, tasks: list[str], dataset: str = "grace"):
    config_entry = MODEL_FACTORY.get(model_type.lower())
    model_prefix = config_entry["prefix"] if config_entry else model_type

    evaluator = GraceEvaluator()
    
    # ** ground truth **
    if dataset == "casimedicos":
        gold_path = settings.CASIMEDICOS_SPLITS["validation"]
    else:
        gold_path = settings.GRACE_SPLITS["validation"]

    for task_id in tasks:
        pred_path = settings.get_prediction_path(model_prefix, model_size, setting, task_id, cleaned=True)
        
        if task_id == "S1":
            evaluator.evaluate_subtask_1(pred_path, gold_path, dataset)
        elif task_id == "S2":
            evaluator.evaluate_subtask_2(pred_path, gold_path, dataset)
        elif task_id == "S3":
            evaluator.evaluate_subtask_3(pred_path, gold_path, dataset)
            
# --- io -------------------------------------------------------------------------

def _save(data: List[Dict[str, Any]], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t >>> Saved results to {out_file.name}")
    
    
def _load(n: int = 4, dataset: str = "grace", balanced_split: bool = True) -> tuple:
    """Loads cases and relations dynamically based on the dataset, with optional unified 50/50 split or randomized examples."""
    
    if dataset == "casimedicos":
        split = settings.CASIMEDICOS_SPLITS
        is_IOB = True
    elif dataset == "unified":
        split = settings.UNIFIED_SPLITS
        is_IOB = False
    else:
        split = settings.GRACE_SPLITS
        is_IOB = False
        
    train_path = split["train"]
    test_path = split["validation"]
    
    if is_IOB:
        train_rel_name = train_path.stem.replace("_ordered", "_relations") + ".jsonl"
        test_rel_name = test_path.stem.replace("_ordered", "_relations") + ".jsonl"
        
        train_rel_path = train_path.with_name(train_rel_name)
        test_rel_path = test_path.with_name(test_rel_name)
        
        logging.info(f"> Loading IOB-formatted cases from {train_path.name} & {test_path.name}")
        test_cases = load_cases_casiMedicos(test_path)    
        train_cases = load_cases_casiMedicos(train_path)
        test_relations = load_relations_casiMedicos(test_rel_path)
        train_relations = load_relations_casiMedicos(train_rel_path)
    else:
        logging.info(f"> Loading GRACE-formatted cases and relations from {train_path.name} and {test_path.name}")
        test_cases = load_cases(test_path)    
        train_cases = load_cases(train_path)
        test_relations = load_relations(test_path)
        train_relations = load_relations(train_path)
    
    train_cases.sort(key=lambda x: str(x.get("id", "")))
    train_relations.sort(key=lambda x: str(x.get("id", "")))
    
    random.seed(42)
    
    # ** EXTENSION: 50/50 split for the unified examples **
    if balanced_split and dataset == "unified":
        logging.info(f"> Enforcing 50/50 GRACE/CASIMEDICOS split for few-shot examples...")
        grace_cases = [c for c in train_cases if c.get("origin", "").upper() == "GRACE"]
        casi_cases = [c for c in train_cases if c.get("origin", "").upper() == "CASIMEDICOS"]
        
        half_n = n // 2
        n_grace = half_n
        n_casi = n - half_n
        
        fs_cases = []
        fs_cases.extend(random.sample(grace_cases, min(n_grace, len(grace_cases))))
        fs_cases.extend(random.sample(casi_cases, min(n_casi, len(casi_cases))))
        
        grace_rels = [r for r in train_relations if r.get("origin", "").upper() == "GRACE"]
        casi_rels = [r for r in train_relations if r.get("origin", "").upper() == "CASIMEDICOS"]
        
        fs_relations = []
        fs_relations.extend(random.sample(grace_rels, min(n_grace, len(grace_rels))))
        fs_relations.extend(random.sample(casi_rels, min(n_casi, len(casi_rels))))
        
        random.shuffle(fs_cases)
        random.shuffle(fs_relations)
    
    # ** RANDOM EXAMPLES **
    else:
        fs_cases = random.sample(train_cases, n) if len(train_cases) >= n else train_cases
        fs_relations = random.sample(train_relations, n) if len(train_relations) >= n else train_relations
    
    return fs_cases, fs_relations, test_cases, test_relations