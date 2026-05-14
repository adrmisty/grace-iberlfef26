# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): few-shot & zero-shot prompting on different models on the CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# may-2026

import argparse
from pathlib import Path
from src.grace.task import run_subtasks, run_global_subtasks, evaluate_subtasks, best_runs_for_s3
from src.grace.post.submit import submit, submit_global, clean
from src.grace.model import MODEL_FACTORY
import src.grace.config as settings
import logging

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="GRACE 'Granular Recognition of Argumentative Clinical Evidence'")
    
    parser.add_argument("--run", action="store_true", help="Run model prompting")
    parser.add_argument("--eval", action="store_true", help="Run metrics calculation")
    parser.add_argument("--clean", action="store_true", help="Run post-processing (clean predictions)")
    parser.add_argument("--submit", action="store_true", help="Run post-processing (compile task submission file)")
    parser.add_argument("--bestrun", action="store_true", help="Run S3 ensemble using another (best-run) model's submission")
    
    parser.add_argument("--model", type=str, default="Qwen", help="Model type: Qwen, MedGemma, Gemini, OpenAI")
    parser.add_argument("--sizes", nargs="+", default=["2B", "4B", "27B"], help="Model sizes")
    parser.add_argument("--settings", nargs="+", default=["zero_shot", "few_shot"], help="Prompt settings")
    parser.add_argument("--tasks", nargs="+", default=["S1", "S2", "S3", "global"], help="Task numbers")

    parser.add_argument("--dataset", type=str, choices=["grace", "casimedicos", "unified", "blind_grace"], default="grace", help="Specify the dataset format for submission compilation (default: grace).")
    parser.add_argument("--n_examples", type=int, default=0, help="Number of examples (only used for few shot learning)")

    parser.add_argument("--other_predictions", type=str, default="", help="Path to the best-run submission file for the ensemble")
    parser.add_argument("--other_model", type=str, default="", help="Name of the best-run model (e.g., qwen3.5)")

    args = parser.parse_args()
    
    config_entry = MODEL_FACTORY.get(args.model.lower())
    model_prefix = config_entry["prefix"] if config_entry else args.model

    # *** run global inference/split subtasks ***
    if args.run:
        if "global" in args.tasks:
            run_global_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, dataset=args.dataset, n_examples=args.n_examples)
        if len([t for t in args.tasks if t != "global"]) > 0:
            run_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, tasks=[t for t in args.tasks if t != "global"], dataset=args.dataset, n_examples=args.n_examples)
    
    # *** run ensemble ***
    if args.bestrun:
        if not args.other_predictions:
            logging.error("\t> (!) You must provide both --other_predictions (and --other_model to extract predictions, not required)")
        else:
            best_runs_for_s3(
                other_predictions=Path(args.other_predictions),
                other_model=args.other_model,
                model_type=args.model,
                sizes=args.sizes,
                prompt_settings=args.settings,
                dataset=args.dataset,
                n_examples=args.n_examples
            )

    # *** post-processing ***
    if args.clean:
        for size in args.sizes:
            for setting in args.settings:
                for task in args.tasks:
                    path = settings.get_prediction_path(model_prefix, size, setting, task, dataset=args.dataset, n_examples=args.n_examples)
                    clean(filepath=path)

    # *** submission format ***
    if args.submit:
        logging.info(f"> Compiling submissions for the GRACE-IBERLEF26 shared task...")
        
        if args.dataset == "grace":
            original_json_path = settings.GRACE_SPLITS["validation"]
        elif args.dataset == "unified":
            original_json_path = settings.UNIFIED_SPLITS["validation"]
        else:
            original_json_path = settings.BLIND_GRACE_SPLITS["validation"]
            
        for size in args.sizes:
            for setting in args.settings:
                out_dir = settings.MODEL_DIR / args.dataset / model_prefix / size / "submission"
                out_dir.mkdir(parents=True, exist_ok=True)

                if "global" in args.tasks:
                    output_path = out_dir / f"{model_prefix}_{size}_{setting}_global_{args.dataset}_{args.n_examples}_submission.json"
                    global_path = settings.get_prediction_path(model_prefix, size, setting, "global", dataset=args.dataset, n_examples=args.n_examples, cleaned=False)
                    submit_global(original_json_path=original_json_path, global_preds_path=global_path, output_path=output_path)
                    
                if any(t in args.tasks for t in ["S1", "S2", "S3"]):
                    output_path = out_dir / f"{model_prefix}_{size}_{setting}_s1s2s3_{args.dataset}_{args.n_examples}_submission.json"
                    s1_path = settings.get_prediction_path(model_prefix, size, setting, task="S1", dataset=args.dataset, n_examples=args.n_examples, cleaned=True)
                    s2_path = settings.get_prediction_path(model_prefix, size, setting, task="S2", dataset=args.dataset, n_examples=args.n_examples, cleaned=True)
                    s3_path = settings.get_prediction_path(model_prefix, size, setting, task="S3", dataset=args.dataset, n_examples=args.n_examples, cleaned=True)
                                                            
                    submit(original_json_path=original_json_path, s1_path=s1_path, s2_path=s2_path, s3_path=s3_path, output_path=output_path)

    # *** specific evaluation ***
    if args.eval:
        for size in args.sizes:
            for setting in args.settings:
                evaluate_subtasks(model_type=args.model, model_size=size, setting=setting, tasks=[t for t in args.tasks if t != "global"], dataset=args.dataset, n_examples=args.n_examples)
                
if __name__ == "__main__":
    main()