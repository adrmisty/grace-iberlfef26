# src/grace/config.py
# ----------------------------------------------------------
# configurations and paths
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

from typing import List, Dict
from pathlib import Path

# --- general path directories and settings ---
LANG = "es"
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
BASE_DATA_DIR: Path = ROOT_DIR / "data"
SPLITS_DATA_DIR: Path = BASE_DATA_DIR / "splits"
GRACE_DATA_DIR: Path = BASE_DATA_DIR / "grace"
UNIFIED_DATA_DIR: Path = BASE_DATA_DIR / "unified"
CASIMEDICOS_DATA_DIR: Path = BASE_DATA_DIR / "casimedicos" / "splits"
MODEL_DIR: Path = ROOT_DIR / "model"


# --- dataset splits ---
HF_REPO: str = "HiTZ/casimedicos-arg"
GRACE_SPLITS: Dict[str, Path] = {
    "train": GRACE_DATA_DIR / "track_2_train.json",
    "validation": GRACE_DATA_DIR / "track_2_dev.json",
}
UNIFIED_SPLITS: Dict[str, Path] = {
    "train": UNIFIED_DATA_DIR / f"train_unified_{LANG}.json",
    "validation": GRACE_DATA_DIR / "track_2_dev.json",
}
CASIMEDICOS_SPLITS: Dict[str, Path] = {
    "train": CASIMEDICOS_DATA_DIR / "train" / f"train_{LANG}_ordered.jsonl",
    "validation": CASIMEDICOS_DATA_DIR / "dev" / f"dev_{LANG}_ordered.jsonl",
    "test": CASIMEDICOS_DATA_DIR / "test" / f"test_{LANG}_ordered.jsonl"
}

def get_prediction_path(model_prefix: str, size: str, setting: str, task: str, dataset: str, output_dir : Path = MODEL_DIR, cleaned: bool = False) -> Path:
    """Central source of truth for prediction file naming conventions."""
    base_name = f"{model_prefix}_{size}_{setting}_{task}_{dataset}"
    ext = ".clean.json" if cleaned else ".json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / f"{base_name}{ext}"