# train.py
# ----------------------------------------------------------------------------------------
# evaluation loop for Qwen3.5 (zero-shot & few-shot) on CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import gc
import json
import torch
import random
import logging
import config as settings
from pathlib import Path
from typing import List, Dict, Any
from model import GraceModel

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")


def evaluate():
    logging.info("> Loading datasets ---")
    
    # ** FEW-SHOT EXAMPLES AND MODEL SET UP **
    train_cases, train_relations, test_cases, test_relations = _load_cases(n=3)
    sizes = ["2B", "4B", "8B"]
    
    for size in sizes:
        logging.info(f"\n========================================================")
        logging.info(f"QWEN3.5-{size} / FEW-SHOT / ZERO-SHOT EVALUATION")
        logging.info(f"========================================================")
        
        evaluator = GraceModel(model_size=size)
        
        for setting in ["zero_shot", "few_shot"]:
            logging.info(f"\n\t >>> [{setting.upper()}] ---")
            
            # SENTENCE RELEVANCE
            s1_examples = train_cases if setting == "few_shot" else None
            s1_results = evaluator.evaluate_subtask_1(test_cases, few_shot_examples=s1_examples)
            _save(s1_results, settings.RESULTS_DIR / f"Qwen_{size}_{setting}_S1.json")
            
            # SPAN EXTRACTION INTO CLAIMS/PREMISES
            s2_examples = train_cases if setting == "few_shot" else None
            s2_results = evaluator.evaluate_subtask_2(test_cases, few_shot_examples=s2_examples)
            _save(s2_results, settings.RESULTS_DIR / f"Qwen_{size}_{setting}_S2.json")
            
            # RELATION EXTRACTION BETWEEN CLAIMS/PREMISES
            s3_examples = train_relations if setting == "few_shot" else None
            s3_results = evaluator.evaluate_subtask_3(test_relations, few_shot_examples=s3_examples)
            _save(s3_results, settings.RESULTS_DIR / f"Qwen_{size}_{setting}_S3.json")
            
        # ** clear memory upon model size change **
        logging.info(f"\t> Clearing Qwen3.5-{size}...")
        del evaluator.model
        del evaluator.tokenizer
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()

# --- io -------------------------------------------------------------------------

def _save(data: List[Dict[str, Any]], out_file: Path):
    """Saves evaluation predictions to disk."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t >>> Saved results to {out_file.name}")
    
def _load_relations(file_path: Path) -> List[Dict[str, Any]]:
    """Flattens the _relations.jsonl file into individual evaluation targets for Subtask 3."""
    relations_list = []
    if not file_path.exists():
        logging.error(f"\t (!) File not found: {file_path}")
        return relations_list
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            record = json.loads(line)
            if isinstance(record, list): record = record[0]
            
            for case_id, rels in record.items():
                for idx, (head, tail, label) in enumerate(rels):
                    relations_list.append({
                        "id": f"{case_id}_{idx}",
                        "case_id": case_id,
                        "head": head,
                        "tail": tail,
                        "label": label
                    })
    return relations_list

def _load_cases(n: int = 3):
    """Loads few-shot examples for prompting after parsing clinical cases and relations."""
    # ** SUBTASKS 1 AND 2: CASE PARSING **
    parse_eval = GraceModel(model_size="2B") 
    test_cases = parse_eval.load_and_parse_data(settings.SPLITS_DATA_DIR / "test" / "test_es_ordered.jsonl")
    train_cases = parse_eval.load_and_parse_data(settings.SPLITS_DATA_DIR / "train" / "train_es_ordered.jsonl")
    
    # ** SUBTASK 3: RELATION LOADING **
    test_relations = _load_relations(settings.SPLITS_DATA_DIR / "test" / "test_es_relations.jsonl")
    train_relations = _load_relations(settings.SPLITS_DATA_DIR / "train" / "train_es_relations.jsonl")
    
    # Select 3 random high-quality examples for few-shot prompting
    random.seed(42) # For reproducibility
    fs_cases = random.sample(train_cases, n) if len(train_cases) >= n else train_cases
    fs_relations = random.sample(train_relations, n) if len(train_relations) >= n else train_relations
    
    del parse_eval.model
    del parse_eval
    torch.cuda.empty_cache()
    gc.collect()
    
    return fs_cases, fs_relations, test_cases, test_relations

