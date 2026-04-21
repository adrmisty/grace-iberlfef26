# main.py
# ------------------------------------------------------------------------------------------------------
# casimedicos relation alignment and multilingual split generation (pre-processing for GRACE IBERLEF26)
# ------------------------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import argparse
import logging
from .relations import RelationAligner
from .splits import SplitGenerator
from .config import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing for CasiMedicos-Arg")
    parser.add_argument("--align", action="store_true", help="Run multi-lingual relation alignment")
    parser.add_argument("--split", action="store_true", help="Run multi-lingual split generation")
    args = parser.parse_args()

    if args.align:
        logging.info(f"[{SOURCE_LANG}] Relation alignment for target languages: {', '.join(TARGET_LANGS)}")
        aligner = RelationAligner(source_lang=SOURCE_LANG)
        
        for lang in TARGET_LANGS:
            for split, relations_path in SPLITS.items():
                if not relations_path.exists():
                    logging.warning(f"\t(!) > Missing {relations_path}... >>> SKIPPED")
                    continue
                    
                aligned_data = aligner.align_split(lang, split, relations_path, MANUAL_FIX_JSON)
                out_path = RELATIONS_DIR / OUTPUT_JSONL.format(lang=lang, jsonl_split=split)
                aligner.save(aligned_data, out_path)

    if args.split:
        ALL_LANGS = [SOURCE_LANG] + TARGET_LANGS
        logging.info(f"Multilingual split generation for languages: {', '.join(ALL_LANGS)}")
        
        generator = SplitGenerator(
            raw_dir=RAW_DATA_DIR,
            relations_dir=RELATIONS_DIR,
            splits_dir=SPLITS_DATA_DIR,
            all_langs=ALL_LANGS
        )
        generator.generate_splits()

if __name__ == "__main__":
    main()