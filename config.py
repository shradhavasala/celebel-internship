"""
Centralized Configuration Module for SmartForecast.
Contains directory paths, dataset schema constants, feature lists, model hyperparameters, and visual themes.
"""

from pathlib import Path
from typing import List, Dict, Any

# =====================================================================
# Project Directories & Paths
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
LOG_FILE = BASE_DIR / "smartforecast.log"

# Ensure essential directories exist when config is imported
for folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ASSETS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Dataset Schema & Target Columns
# =====================================================================
DATE_COL: str = "date"
STORE_COL: str = "store"
ITEM_COL: str = "item"
TARGET_COL: str = "sales"

# Required columns for training CSV validation
REQUIRED_TRAIN_COLS: List[str] = [DATE_COL, STORE_COL, ITEM_COL, TARGET_COL]
REQUIRED_TEST_COLS: List[str] = [DATE_COL, STORE_COL, ITEM_COL]


# =====================================================================
# Feature Engineering Configuration
# =====================================================================
# Lag steps to generate (in days)
LAG_DAYS: List[int] = [1, 7, 14, 30]

# Rolling windows for mean, std, min, max (in days)
ROLLING_WINDOWS: List[int] = [7, 14, 30]

# Categorical columns to encode
GROUP_COLS: List[str] = [STORE_COL, ITEM_COL]


# =====================================================================
# Model Training & Cross-Validation Defaults
# =====================================================================
DEFAULT_CV_SPLITS: int = 5
DEFAULT_EARLY_STOPPING_ROUNDS: int = 30
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_FORECAST_HORIZON: int = 30  # days


# =====================================================================
# Model Hyperparameters
# =====================================================================
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LightGBM": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    },
    "XGBoost": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist"
    },
    "CatBoost": {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": DEFAULT_RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1
    },
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": -1
    }
}


# =====================================================================
# Visualization Theme & Color Palettes
# =====================================================================
PLOTLY_THEME: str = "plotly_white"
COLOR_PALETTE: List[str] = [
    "#3B82F6",  # Vibrant Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber Yellow
    "#EF4444",  # Coral Red
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#64748B"   # Slate Gray
]

COLOR_ACTUAL = "#3B82F6"
COLOR_PREDICTED = "#EF4444"
COLOR_FORECAST = "#10B981"
