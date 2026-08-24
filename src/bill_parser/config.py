""""Single source of truth for paths and model IDs"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
  _d.mkdir(parents=True, exist_ok=True)

BASE_MODEL_ID = "naver-clova-ix/donut-base"
FINETUNED_REFERENCE_ID = 'naver-clova-ix/donut-base-finetuned-cord-v2'
DATASET_ID = "naver-clova-ix/cord-v2"


SEED = 42