# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): relation alignment, Qwen few-shot & zero-shot prompting and evaluation on CasiMedicos-Arg
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
    
    parser.add_argument("--align", action="store_true", help="Run multi-lingualrelation alignment on CasiMedicos-Arg")
    parser.add_argument("--run", action="store_true", help="Run model prompting")
    parser.add_argument("--evaluate", action="store_true", help="Run metrics calculation")
    
    parser.add_argument("--post", action="store_true", help="Run post-processing (clean predictions)")
    
    parser.add_argument("--sizes", nargs="+", default=["2B", "4B", "27B"], help="Model sizes (2B, 4B or 8B)")
    parser.add_argument("--settings", nargs="+", default=["zero_shot", "few_shot"], help="Prompt settings")
    parser.add_argument("--tasks", nargs="+", default=["S1", "S2", "S3"], help="Task number")

    args = parser.parse_args()

    if args.align:
        align()

    if args.run:
        run_subtasks(sizes=args.sizes, prompt_settings=args.settings)

    if args.post:
        for size in args.sizes:
            for setting in args.settings:
                for task in args.tasks:
                    clean(filepath=f"{size}/Qwen_{size}_{setting}_{task}.json")

    if args.evaluate:
        for size in args.sizes:
            for setting in args.settings:
                evaluate_subtasks(model_size=size, setting=setting)

if __name__ == "__main__":
    main()