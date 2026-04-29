# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): few-shot & zero-shot prompting on different models on the CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import argparse
from .task import run_subtasks, run_global_subtasks, evaluate_subtasks
from .post import clean
from .submit import submit, submit_global
from .model import MODEL_FACTORY
import src.config as settings
import logging

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="GRACE 'Granular Recognition of Argumentative Clinical Evidence'")
    
    parser.add_argument("--run", action="store_true", help="Run model prompting")
    parser.add_argument("--eval", action="store_true", help="Run metrics calculation")
    parser.add_argument("--post", action="store_true", help="Run post-processing (clean predictions)")
    parser.add_argument("--submit", action="store_true", help="Run post-processing (compile task submission file)")
    
    parser.add_argument("--model", type=str, default="Qwen", help="Model type: Qwen, MedGemma, Gemini, OpenAI")
    parser.add_argument("--sizes", nargs="+", default=["2B", "4B", "27B"], help="Model sizes")
    parser.add_argument("--settings", nargs="+", default=["zero_shot", "few_shot"], help="Prompt settings")
    parser.add_argument("--tasks", nargs="+", default=["S1", "S2", "S3", "global"], help="Task numbers")

    parser.add_argument("--dataset", type=str, choices=["grace", "casimedicos", "unified"], default="grace", help="Specify the dataset format for submission compilation (default: grace).")

    args = parser.parse_args()
    
    config_entry = MODEL_FACTORY.get(args.model.lower())
    model_prefix = config_entry["prefix"] if config_entry else "Qwen"

    if args.run:
        if "global" in args.tasks:
            run_global_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, dataset=args.dataset)
        else:
            run_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, tasks=args.tasks, dataset=args.dataset)
        
    if args.post:
        for size in args.sizes:
            for setting in args.settings:
                for task in args.tasks:
                    path = settings.get_prediction_path(model_prefix, size, setting, task, dataset=args.dataset)
                    clean(filepath=path)

    if args.submit:
        logging.info(f"> Compiling submissions for dataset format: {args.dataset.upper()}")
        
        if args.dataset == "grace":
            original_json_path = settings.GRACE_SPLITS["validation"]
        else:
            original_json_path = settings.UNIFIED_SPLITS["validation"]

        for size in args.sizes:
            for setting in args.settings:
                out_dir = settings.MODEL_DIR / args.dataset / model_prefix / size
                output_path = out_dir / f"{model_prefix}_{size}_{setting}_submission.json"

                if "global" in args.tasks:
                    submit_global(original_json_path,
                                settings.get_prediction_path(output_dir=out_dir, model_prefix=model_prefix, size=size, setting=setting, task="global", dataset=args.dataset, cleaned=False),
                                output_path)
                    continue
                else:
                    s1_path = settings.get_prediction_path(model_prefix, size, setting, task="S1", dataset=args.dataset, output_dir=out_dir, cleaned=True)
                    s2_path = settings.get_prediction_path(model_prefix, size, setting, task="S2", dataset=args.dataset, output_dir=out_dir, cleaned=True)
                    s3_path = settings.get_prediction_path(model_prefix, size, setting, task="S3", dataset=args.dataset, output_dir=out_dir, cleaned=True)
                                        
                    submit(original_json_path, s1_path, s2_path, s3_path, output_path=output_path)

                    # CASIMEDICOS
                    """
                    if args.dataset == "casimedicos":
                        cases_path = settings.CASIMEDICOS_SPLITS["validation"]
                        
                        rels_name = cases_path.stem.replace("_ordered", "_relations") + ".jsonl"
                        rels_path = cases_path.with_name(rels_name)
                        
                        submit_casiMedicos(cases_path, rels_path, s1_path, s2_path, s3_path, output_path)
                    """

    if args.eval:
        for size in args.sizes:
            for setting in args.settings:
                evaluate_subtasks(
                    model_type=args.model, 
                    model_size=size, 
                    setting=setting, 
                    tasks=args.tasks,
                    dataset=args.dataset 
                )
                
if __name__ == "__main__":
    main()