# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): few-shot & zero-shot prompting on different models on the CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import argparse
from src.grace.task import run_subtasks, evaluate_subtasks
from src.grace.post import clean
from src.grace.model import MODEL_FACTORY
import src.config as settings
import logging

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="GRACE 'Granular Recognition of Argumentative Clinical Evidence'")
    
    parser.add_argument("--run", action="store_true", help="Run model prompting")
    parser.add_argument("--eval", action="store_true", help="Run metrics calculation")
    parser.add_argument("--post", action="store_true", help="Run post-processing (clean predictions)")
    
    parser.add_argument("--model", type=str, default="Qwen", help="Model type: Qwen, MedGemma, Gemini, OpenAI")
    parser.add_argument("--sizes", nargs="+", default=["2B", "4B", "27B"], help="Model sizes")
    parser.add_argument("--settings", nargs="+", default=["zero_shot", "few_shot"], help="Prompt settings")
    parser.add_argument("--tasks", nargs="+", default=["S1", "S2", "S3"], help="Task numbers")

    args = parser.parse_args()
    
    config_entry = MODEL_FACTORY.get(args.model.lower())
    model_prefix = config_entry["prefix"] if config_entry else "Qwen"

    if args.run:
        run_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, tasks=args.tasks)

    if args.post:
        for size in args.sizes:
            for setting in args.settings:
                for task in args.tasks:
                    path = settings.get_prediction_path(model_prefix, size, setting, task)
                    clean(filepath=path)

    if args.evaluate:
        for size in args.sizes:
            for setting in args.settings:
                evaluate_subtasks(model_type=args.model, model_size=size, setting=setting, tasks=args.tasks)

if __name__ == "__main__":
    main()