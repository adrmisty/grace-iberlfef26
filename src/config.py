# src/config.py
# ----------------------------------------------------------
# configurations and paths
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

from typing import List, Dict
from pathlib import Path

# --- general paths and settings ---

ROOT_DIR: Path = Path(__file__).resolve().parent.parent

HF_REPO: str = "HiTZ/casimedicos-arg"

BASE_DATA_DIR: Path = ROOT_DIR / "data"
SPLITS_DATA_DIR: Path = BASE_DATA_DIR / "splits"
GRACE_DATA_DIR: Path = BASE_DATA_DIR / "grace"
MODEL_DIR: Path = ROOT_DIR / "model" / "grace"
RELATIONS_DIR: Path = BASE_DATA_DIR / "relations"

GRACE_SPLITS: Dict[str, Path] = {
    "train": GRACE_DATA_DIR / "track_2_train.json",
    "validation": GRACE_DATA_DIR / "track_2_dev.json",
}


# --- relation alignment settings ---

SOURCE_LANG: str = "en"
TARGET_LANGS: List[str] = ["es", "fr", "it"]

SPLITS: Dict[str, Path] = {
    "train": RELATIONS_DIR / SOURCE_LANG / "train_relations.jsonl",
    "validation": RELATIONS_DIR / SOURCE_LANG / "validation_relations.jsonl",
    "test": RELATIONS_DIR / SOURCE_LANG / "test_relations.jsonl"
}

OUTPUT_JSONL: str = "{lang}/{jsonl_split}_relations.jsonl"
MANUAL_FIX_JSON: Path = RELATIONS_DIR / "fix_relations.json"

def get_prediction_path(model_prefix: str, size: str, setting: str, task: str, cleaned: bool = False) -> Path:
    """Central source of truth for prediction file naming conventions."""
    base_name = f"{model_prefix}_{size}_{setting}_{task}"
    ext = ".clean.json" if cleaned else ".json"
    output_dir = MODEL_DIR / size
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / f"{base_name}{ext}"