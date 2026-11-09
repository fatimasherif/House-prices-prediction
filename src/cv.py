"""Shared settings and helpers used by every notebook."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# One seed for the whole project; pass it as random_state everywhere.
SEED = 42

# This file lives in src/, so the project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_PATH = ROOT / "results.csv"


def _json_default(value):
    """Convert NumPy scalar values into JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def log_result(
    model_name,
    feature_set,
    params,
    fold_scores,
    notes="",
    phase=None,
    n_features=None,
):
    """Append one row to results.csv: one row per model run."""
    scores = np.asarray(list(fold_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("fold_scores must contain at least one score.")
    if not np.isfinite(scores).all():
        raise ValueError("fold_scores must contain only finite scores.")
    if (scores < 0).any():
        raise ValueError("fold_scores must contain non-negative RMSE values.")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "model": model_name,
        "feature_set": feature_set,
        "params": json.dumps(params, sort_keys=True, default=_json_default),
        "cv_rmse_mean": float(np.mean(scores)),
        "cv_rmse_std": float(np.std(scores, ddof=0)),
        "fold_scores": json.dumps([round(score, 5) for score in scores]),
        "n_features": n_features,
        "notes": notes,
    }
    pd.DataFrame([row]).to_csv(
        RESULTS_PATH,
        mode="a",
        header=not RESULTS_PATH.exists(),
        index=False,
    )