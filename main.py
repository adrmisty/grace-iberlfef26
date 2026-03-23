# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): relation alignment, Qwen & MedGemma few-shot & zero-shot prompting on CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import argparse
from relations import align
from task import run_subtasks, evaluate_subtasks
from post import clean
import logging

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="GRACE 'Granular Recognition of Argumentative Clinical Evidence' / dataset: CasiMedicos-Arg")
    
    parser.add_argument("--align", action="store_true", help="Run multi-lingual relation alignment on CasiMedicos-Arg")
    parser.add_argument("--run", action="store_true", help="Run model prompting")
    parser.add_argument("--evaluate", action="store_true", help="Run metrics calculation")
    parser.add_argument("--post", action="store_true", help="Run post-processing (clean predictions)")
    
    parser.add_argument("--model", type=str, default="Qwen", help="Model type: Qwen or MedGemma")
    parser.add_argument("--sizes", nargs="+", default=["2B", "4B", "27B"], help="Model sizes (e.g., 2B 4B 8B)")
    parser.add_argument("--settings", nargs="+", default=["zero_shot", "few_shot"], help="Prompt settings")
    parser.add_argument("--tasks", nargs="+", default=["S1", "S2", "S3"], help="Task numbers")

    args = parser.parse_args()
    
    model_prefix = "MedGemma" if args.model.lower() == "medgemma" else "Qwen"

    if args.align:
        align()

    if args.run:
        run_subtasks(model_type=args.model, sizes=args.sizes, prompt_settings=args.settings, tasks=args.tasks)

    if args.post:
        for size in args.sizes:
            for setting in args.settings:
                for task in args.tasks:
                    clean(filepath=f"{size}/{model_prefix}_{size}_{setting}_{task}.json")

    if args.evaluate:
        for size in args.sizes:
            for setting in args.settings:
                evaluate_subtasks(model_type=args.model, model_size=size, setting=setting, tasks=args.tasks)

if __name__ == "__main__":
    main()