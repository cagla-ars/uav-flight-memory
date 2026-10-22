from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CSV_ROOT = RAW_DIR / "csv_files_annotated"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
MODELS_DIR = RESULTS_DIR / "models"

HF_DATASET = "aykutkabaoglu/UAV-SEAD_State_Estimation_Anomaly_Dataset"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

TARGET_CLASSES = (
    "Normal",
    "External Position",
    "Altitude",
    "Mechanical",
    "Global Position",
)
EXCLUDED_CLASSES = ("Uncategorized",)
RARE_CLASSES = ("Altitude", "Mechanical", "Global Position")

LOCAL_POSITION_COLUMNS = (
    "vehicle_local_position.x",
    "vehicle_local_position.y",
    "vehicle_local_position.z",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
)
FEATURE_STATS = ("mean", "std", "min", "max")
K_VALUES = (1, 3, 5, 7, 9)
DISTANCE_METRICS = ("euclidean", "manhattan")
SPLIT_TRAIN_RATIOS = (0.8, 0.9)

DEFAULT_SEED = 42
DEFAULT_WINDOW_SECONDS = 20
DEFAULT_MIN_ANOMALY_COVERAGE = 0.10
DEFAULT_NORMAL_LIMIT = 120
DEFAULT_EXTERNAL_LIMIT = 160

