# main.py
# --------------------------------------------------------------------------------------------------------------
# (GRACE / IBERLEF26): few-shot & zero-shot prompting on different models on the CasiMedicos-Arg
# ---------------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import argparse
import logging
from .relations import RelationAligner
import src.config as settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing for CasiMedicos-Arg")
    parser.add_argument("--align", action="store_true", help="Run multi-lingual relation alignment")
    args = parser.parse_args()

    if args.align:
        logging.info(f"[{settings.SOURCE_LANG}] Relation alignment for target languages: {', '.join(settings.TARGET_LANGS)}")
        aligner = RelationAligner(source_lang=settings.SOURCE_LANG)
        
        for lang in settings.TARGET_LANGS:
            for split, relations_path in settings.SPLITS.items():
                if not relations_path.exists():
                    logging.warning(f"\t(!) > Missing {relations_path}... >>> SKIPPED")
                    continue
                    
                aligned_data = aligner.align_split(lang, split, relations_path, settings.MANUAL_FIX_JSON)
                out_path = settings.RELATIONS_DIR / settings.OUTPUT_JSONL.format(lang=lang, jsonl_split=split)
                aligner.save(aligned_data, out_path)

if __name__ == "__main__":
    main()