"""Shared settings, the validation protocol and results logging.

Every model in the project is scored with evaluate(), so all of them use the
same folds, the same metric and the same results table.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

# One seed for the whole project; pass it as random_state everywhere
SEED = 42

# This file lives in src/, so the project root is one level up
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_PATH = ROOT / "results.csv"

# The validation protocol, fixed for the whole project.
# Same splitter object everywhere -> every model sees exactly the same 5 folds.
N_SPLITS = 5
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
# scikit-learn always maximizes a score, so RMSE comes back with a minus sign
SCORING = "neg_root_mean_squared_error"

# Fixed column layout of results.csv
RESULT_COLUMNS = [
    "timestamp", "phase", "model", "feature_set", "params", "n_features",
    "cv_rmse_mean", "cv_rmse_std", "fold_scores", "notes",
]


def log_result(model_name, feature_set, params, fold_scores, notes="", phase=None, n_features=None):
    """Save one model run to results.csv (one row per model + feature set).

    Running the same model on the same feature set again replaces its old row,
    so "Run All" never creates duplicates.
    """
    scores = np.asarray(list(fold_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("fold_scores must contain at least one score.")
    if not np.isfinite(scores).all():
        raise ValueError("fold_scores must contain only finite scores.")
    if (scores < 0).any():
        raise ValueError("RMSE can't be negative: did you forget the minus sign "
                         "in front of cross_val_score with neg_root_mean_squared_error?")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "model": model_name,
        "feature_set": feature_set,
        # default=str handles numpy integers and other non-JSON types
        "params": json.dumps(params, sort_keys=True, default=str),
        "n_features": n_features,
        "cv_rmse_mean": round(float(scores.mean()), 5),
        "cv_rmse_std": round(float(scores.std(ddof=0)), 5),
        "fold_scores": json.dumps([round(float(s), 5) for s in scores]),
        "notes": notes,
    }
    new_row = pd.DataFrame([row], columns=RESULT_COLUMNS)

    # An empty file (e.g. contents deleted by hand) counts as no file
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        table = pd.read_csv(RESULTS_PATH)
        same_run = (table["model"] == model_name) & (table["feature_set"] == feature_set)
        table = pd.concat([table.loc[~same_run, RESULT_COLUMNS], new_row], ignore_index=True)
    else:
        table = new_row
    table.to_csv(RESULTS_PATH, index=False)


def evaluate(model, X, y, name, feature_set, params=None, notes="", phase=None,
             n_features=None, log=True):
    """Score a model on the shared folds, print mean ± std and log it.

    Returns the 5 fold RMSEs (on log price) so models can be compared fold by fold.
    """
    scores = -cross_val_score(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    print(f"{name} [{feature_set}]: RMSE {scores.mean():.4f} ± {scores.std():.4f} "
          f"| folds {np.round(scores, 4).tolist()}")
    if log:
        log_result(name, feature_set, params or {}, scores,
                   notes=notes, phase=phase, n_features=n_features)
    return scores


def show_results():
    """The results table, best model first."""
    table = pd.read_csv(RESULTS_PATH)
    cols = ["phase", "model", "feature_set", "n_features", "cv_rmse_mean", "cv_rmse_std", "notes"]
    table["notes"] = table["notes"].fillna("")
    return table.sort_values("cv_rmse_mean")[cols].reset_index(drop=True)


def oof_predict(model, X, y, train_keep=None):
    """Out-of-fold predictions on the shared folds.

    Every house is predicted by a model that never saw it. train_keep is an
    optional boolean mask of rows allowed in the TRAINING folds (e.g. to drop
    outliers from training while still predicting them). The folds are always
    the shared ones on the full data, so results stay comparable.
    """
    n = len(y)
    train_keep = np.ones(n, dtype=bool) if train_keep is None else np.asarray(train_keep, dtype=bool)
    pred = np.empty(n, dtype=float)
    for tr_idx, va_idx in CV.split(X):
        tr_idx = tr_idx[train_keep[tr_idx]]
        fitted = clone(model).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred[va_idx] = fitted.predict(X.iloc[va_idx])
    return pred


def fold_rmse(y, pred, score_keep=None):
    """RMSE per shared fold, optionally counting only the houses in score_keep."""
    y = np.asarray(y, dtype=float)
    score_keep = np.ones(len(y), dtype=bool) if score_keep is None else np.asarray(score_keep, dtype=bool)
    scores = []
    for _, va_idx in CV.split(y):
        va_idx = va_idx[score_keep[va_idx]]
        scores.append(np.sqrt(np.mean((y[va_idx] - pred[va_idx]) ** 2)))
    return np.array(scores)