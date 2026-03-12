# config.py
# ----------------------------------------------------------
# configurations and paths
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from typing import List, Dict
from pathlib import Path

HF_REPO: str = "HiTZ/casimedicos-arg"

BASE_DATA_DIR: Path = Path("data")
SPLITS_DATA_DIR: Path = Path("splits")
MODEL_DIR: Path = Path("model")

RELATIONS_DIR: Path = BASE_DATA_DIR / "relations"

SOURCE_LANG: str = "en"
TARGET_LANGS: List[str] = ["es", "fr", "it"]

SPLITS: Dict[str, Path] = {
    "train": RELATIONS_DIR / SOURCE_LANG / "train_relations.jsonl",
    "validation": RELATIONS_DIR / SOURCE_LANG / "validation_relations.jsonl",
    "test": RELATIONS_DIR / SOURCE_LANG / "test_relations.jsonl"
}

OUTPUT_JSONL: str = "{lang}/{jsonl_split}_relations.jsonl"
MANUAL_FIX_JSON: Path = RELATIONS_DIR / "fix_relations.json"