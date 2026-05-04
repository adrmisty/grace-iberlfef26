# task.py
# ---------------------------------------------------------------------------------------------
# zero-shot and few-shot prompting and pipeline run for all 3 GRACE subtasks
# extension for dataset splits
# extension2 for global inference
# ---------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import gc
import torch
import logging

import src.grace.config as settings
from src.grace import case
from src.grace.eval.metric import GraceEvaluator
from src.grace.model import MODEL_FACTORY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt='%H:%M:%S')

def run_global_subtasks(model_type: str, sizes: list[str], prompt_settings: list[str], dataset: str = "grace", balanced_split: bool = True, n_examples: int = 4, lang_code: str = "es"):
    """Runs the prompting pipeline for ALL subtasks in a single one-step inference."""    
    
    train_cases, train_relations, test_cases, test_relations = case.load_dataset(dataset=dataset)
        
    for size in sizes:
        config_entry = MODEL_FACTORY.get(model_type.lower())
        model_prefix = config_entry["prefix"] if config_entry else model_type
        model = MODEL_FACTORY[model_type.lower()]["class"](size)
            
        logging.info(f"\n========================================================")
        logging.info(f"{model_prefix.upper()}-{size} / GLOBAL INFERENCE (dataset: {dataset.upper()}) / ")
        logging.info(f"========================================================")
        
        for setting in prompt_settings:
            logging.info(f"\n\t >>> [{setting.upper()} ({n_examples} EXAMPLES)] ---")
            
            # ** setting: few/zero **
            if setting == "few_shot":
                fs_cases, fs_rels = case.sample_few_shot(
                    train_cases, train_relations, n=n_examples, dataset=dataset, balanced_split=balanced_split
                )
            else:
                fs_cases, fs_rels = None, None

            run_func = getattr(model, "run_global", None)
            
            if not run_func:
                raise NotImplementedError(f"\t> (!) Model {model_type} has not implemented global inference yet")

            # (S1+S2+S3)
            results = run_func(
                test_cases, 
                few_shot_examples=fs_cases, 
                example_relations=fs_rels, 
                lang=lang_code
            )
            
            # ** global save **
            out_path = settings.get_prediction_path(model_prefix, size, setting, "global", dataset, n_examples)
            case.save_predictions(results, out_path)
            
        logging.info(f"\t> Clearing {model_prefix}-{size}...")
        del model
        torch.cuda.empty_cache()
        gc.collect()


def run_subtasks(model_type: str, sizes: list[str], prompt_settings: list[str], tasks: list[str] = ["S1", "S2", "S3"], dataset: str = "grace", balanced_split: bool = True, n_examples: int = 4):
    """Runs the prompting pipeline for all specified subtasks and settings for a given model."""    
    
    train_cases, train_relations, test_cases, test_relations = case.load_dataset(dataset=dataset)

    
    for size in sizes:
        config_entry = MODEL_FACTORY.get(model_type.lower())
        model_prefix = config_entry["prefix"] if config_entry else model_type
        model = MODEL_FACTORY[model_type.lower()]["class"](size)
            
        logging.info(f"\n========================================================")
        logging.info(f"{model_prefix.upper()}-{size} / SUBTASK INFERENCE (dataset: {dataset.upper()}) / ")
        logging.info(f"========================================================")
        
        for setting in prompt_settings:
<<<<<<< HEAD
            logging.info(f"\t>>> [{setting.upper()}] ---")
            
            if setting == "few_shot":
                fs_cases, fs_rels = case.sample_few_shot(
                    train_cases, train_relations, n=n_examples, dataset=dataset, balanced_split=balanced_split
                )
            else:
                fs_cases, fs_rels = None, None
=======
            logging.info(f"\n\t >>> [{setting.upper()} ({n_examples} EXAMPLES)] ---")
>>>>>>> 890a191742ef9736c0e703054d2fa1323720c6a8
            
            for task_id in tasks:
                run_func = getattr(model, f"run_subtask_{task_id[-1]}")
                
                if task_id == "S3":
                    data = test_relations
                    examples = fs_rels
                else:
                    data = test_cases
                    examples = fs_cases

                results = run_func(data, few_shot_examples=examples, lang="es")
                
                out_path = settings.get_prediction_path(model_prefix, size, setting, task_id, dataset, n_examples)
                case.save_predictions(results, out_path)
            
        logging.info(f"\t> Clearing {model_prefix}-{size}...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
                

def evaluate_subtasks(model_type: str, model_size: str, setting: str, tasks: list[str], dataset: str = "grace", n_examples: int = 4):
    config_entry = MODEL_FACTORY.get(model_type.lower())
    model_prefix = config_entry["prefix"] if config_entry else model_type

    evaluator = GraceEvaluator()
    
    # ** ground truth **
    if dataset == "casimedicos":
        gold_path = settings.CASIMEDICOS_SPLITS["validation"]
    else:
        gold_path = settings.GRACE_SPLITS["validation"]

    for task_id in tasks:
        pred_path = settings.get_prediction_path(model_prefix, model_size, setting, task_id, dataset, n_examples, cleaned=True)
        
        if task_id == "S1":
            evaluator.evaluate_subtask_1(pred_path, gold_path, dataset)
        elif task_id == "S2":
            evaluator.evaluate_subtask_2(pred_path, gold_path, dataset)
        elif task_id == "S3":
<<<<<<< HEAD
            evaluator.evaluate_subtask_3(pred_path, gold_path, dataset)
=======
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
        logging.info(f"> Enforcing 50/50 GRACE/CASIMEDICOS {n} split for few-shot examples...")
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
>>>>>>> 890a191742ef9736c0e703054d2fa1323720c6a8
