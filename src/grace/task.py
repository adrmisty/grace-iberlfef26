# task.py
# ---------------------------------------------------------------------------------------------
# zero-shot and few-shot prompting and pipeline run for all 3 GRACE subtasks
# extension for dataset splits
# extension2 for global inference
# ensemble extension for external S3 cross-evaluation
# ---------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import gc
import json
import torch
import copy
import logging
from pathlib import Path

import src.grace.config as settings
from src.grace import case
from src.grace.eval.metric import GraceEvaluator
from src.grace.model import MODEL_FACTORY
from src.grace.post.submit import _parse_json, _extract_s3_label, _get_raw_text

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
            
            if setting == "few_shot":
                fs_cases, fs_rels = case.sample_few_shot(train_cases, train_relations, n=n_examples, dataset=dataset, balanced_split=balanced_split)
            else:
                fs_cases, fs_rels = None, None

            run_func = getattr(model, "run_global", None)
            if not run_func:
                raise NotImplementedError(f"\t> (!) Model {model_type} has not implemented global inference yet")

            results = run_func(
                test_cases, 
                few_shot_examples=fs_cases, 
                example_relations=fs_rels, 
                lang=lang_code
            )
            
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
            logging.info(f"\t>>> [{setting.upper()}] ---")
            
            if setting == "few_shot":
                fs_cases, fs_rels = case.sample_few_shot(train_cases, train_relations, n=n_examples, dataset=dataset, balanced_split=balanced_split)
            else:
                fs_cases, fs_rels = None, None
            
            for task_id in tasks:
                if task_id == "global":
                    continue
                
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


# --------------------------------------- best runs (S3 generation)


def best_runs_for_s3(other_predictions: Path, other_model: str = "", model_type: str = "OpenAI", sizes: list[str] = ["gpt-4o-mini"], prompt_settings: list[str] = ["few_shot"], dataset: str = "grace", balanced_split: bool = True, n_examples: int = 4, lang_code: str = "es"):
    """Takes an existing submission file [off of the best-runs list], pairs its predicted Premises and Claims, and feeds them to another model to gather S3 predictions.
    """
    if not other_predictions.exists():
        logging.error(f"\t> (!) Submission file not found: {other_predictions}")
        return

    # ** source model
    dataset = None
    if not other_model:
        filename = other_predictions.stem # Removes .json
        if "-" in filename:
            name = filename.split("-", 1)
            dataset = name[0]
            other_model = name[-1]
        else:
            other_model = filename

    with open(other_predictions, 'r', encoding='utf-8') as f:
        other_cases = json.load(f)
        if isinstance(other_cases, dict): other_cases = [other_cases]

    # dataset and few_shot examples if any
    train_cases, train_relations, _, _ = case.load_dataset(dataset=dataset)

    for size in sizes:
        config_entry = MODEL_FACTORY.get(model_type.lower())
        model_prefix = config_entry["prefix"] if config_entry else model_type
        model = MODEL_FACTORY[model_type.lower()]["class"](size)
        
        logging.info(f"\n========================================================")
        logging.info(f"BEST_RUN: [S1/S2: {other_model.upper()}] + [S3: {model_prefix.upper()}-{size}]")
        logging.info(f"========================================================")

        for setting in prompt_settings:
            logging.info(f"\n\t >>> [{setting.upper()} ({n_examples} EXAMPLES)] ---")
            
            cases = copy.deepcopy(other_cases)
            
            candidate_relations = []
            case_to_candidates = {str(c["id"]): [] for c in cases}
            
            for c_case in cases:
                case_id = str(c_case["id"])
                raw_text = _get_raw_text(c_case)
                entities = c_case.get("predictions", {}).get("entities", [])
                
                premises = [e for e in entities if e.get("type") == "Premise"]
                claims = [e for e in entities if e.get("type") == "Claim"]
                
                for p in premises:
                    for c in claims:
                        cand_id = f"cand_{case_id}_{p['id']}_{c['id']}"
                        cand_obj = {
                            "id": cand_id,
                            "case_id": case_id,
                            "text": raw_text,
                            "head": p["text"],
                            "tail": c["text"],
                            "arg1_id": p["id"],
                            "arg2_id": c["id"]
                        }
                        candidate_relations.append(cand_obj)
                        case_to_candidates[case_id].append(cand_obj)

            if not candidate_relations:
                logging.warning(f"\t> (!) No valid entities found in {other_predictions.name}.")
                continue

            if setting == "few_shot":
                _, fs_rels = case.sample_few_shot(train_cases, train_relations, n=n_examples, dataset=dataset, balanced_split=balanced_split)
            else:
                fs_rels = None

            # ** subtask 3 run **
            results = model.run_subtask_3(candidate_relations, few_shot_examples=fs_rels, lang=lang_code)
            res_map = {str(res["id"]): res.get("prediction", res.get("predictions", "")) for res in results}

            for c_case in cases:
                case_id = str(c_case["id"])
                if "predictions" not in c_case:
                    c_case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}
                    
                # > enrich with model of origin metadata
                if dataset:
                    c_case["predictions"]["ORIGIN"] = {
                        "S1_S2": other_model,
                        "S3": f"{model_prefix}-{size}",
                        "dataset": f"{dataset.upper()}"
                    }
                else:
                    c_case["predictions"]["ORIGIN"] = {
                        "S1_S2": other_model,
                        "S3": f"{model_prefix}-{size}"
                    }

                new_relations = []
                for cand in case_to_candidates.get(case_id, []):
                    raw_pred = res_map.get(cand["id"], "")
                    p_obj = _parse_json(raw_pred)
                    
                    label = _extract_s3_label(p_obj).strip().capitalize()
                    # ** reconstruct submission file with new predictions **
                    if label in ["Support", "Attack"]:
                        new_relations.append({
                            "id": f"ens_rel_{case_id}_{len(new_relations)+1}",
                            "arg1_id": cand["arg1_id"],
                            "arg2_id": cand["arg2_id"],
                            "relation_type": label
                        })

                c_case["predictions"]["relations"] = new_relations

            out_dir = other_predictions.parent / "ensemble"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_filename = f"bestrun_{other_model}_{model_prefix}-{size}_{setting}_{dataset}.json"
            out_path = out_dir / out_filename

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

            logging.info(f"\t>>> Successfully saved [S3 best run ensemble] to: {out_path.name}")

        logging.info(f"\t> Clearing {model_prefix}-{size}...")
        del model
        torch.cuda.empty_cache()
        gc.collect()

# --------------------------------------- specific evaluation (not GRACE TASK)

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
            evaluator.evaluate_subtask_3(pred_path, gold_path, dataset)