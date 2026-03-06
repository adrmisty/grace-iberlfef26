# main.py
# ----------------------------------------------------------------------------------------
# relation mapping and alignment for CasiMedicos-Arg (GRACE / IBERLEF26)
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import logging
import config as settings
from relations import RelationAligner

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"[{settings.SOURCE_LANG}] relation alignment for target languages: {', '.join(settings.TARGET_LANGS)}")
    
    aligner = RelationAligner(source_lang=settings.SOURCE_LANG)
    
    for lang in settings.TARGET_LANGS:
        for split, relations_path in settings.SPLITS.items():
            if not relations_path.exists():
                logging.warning(f"\t(!) > Missing {relations_path}... >>> SKIPPED RELATION")
                continue
                
            aligned_data = aligner.align_split(lang, split, relations_path, settings.MANUAL_FIX_JSON)
            
            aligner.save(aligned_data, settings.RELATIONS_DIR / settings.OUTPUT_JSONL.format(lang=lang, jsonl_split=split))           

if __name__ == "__main__":
    main()