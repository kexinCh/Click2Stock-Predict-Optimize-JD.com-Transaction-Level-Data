from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

OUTPUTS_DIR = ROOT_DIR / "outputs"
EXPLORATION_OUTPUTS_DIR = OUTPUTS_DIR / "exploration"
MODEL_OUTPUTS_DIR = OUTPUTS_DIR / "models" / "warehouse_demand"
REGION_MODEL_OUTPUTS_DIR = OUTPUTS_DIR / "models" / "region_demand"
BASELINE_OUTPUTS_DIR = OUTPUTS_DIR / "models" / "baseline_comparison"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"
REGION_PROCESSED_DIR = PROCESSED_DATA_DIR / "byregion"


def resolve_path(path_str: str | Path, *, base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
