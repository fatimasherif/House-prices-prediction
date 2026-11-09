# House Prices: Advanced Regression Techniques

This project explores the Kaggle House Prices dataset, from data inspection and
feature engineering through model validation and ensembling.

## Project layout

```text
data/
  raw/          Original Kaggle files; do not edit these files
  processed/    Optional outputs created during cleaning
notebooks/      Ordered exploration, modeling, and validation notebooks
src/
  features.py   Cleaning and feature-engineering code
  cv.py         Shared cross-validation and scoring helpers
reports/
  figures/      Saved plots
results.csv     One row per model run
NOTES.md        Working notes and project conventions
requirements.txt
```

The existing notebooks are numbered in workflow order. Add new notebooks using
the next available number rather than replacing or renaming existing work.

## Decision log

- Python version: 3.14.5, the project's active environment; all planned
  libraries, including CatBoost, SHAP, and LightGBM, import successfully.
- Shared random seed: `SEED = 42`, so splits and randomized models are
  reproducible and comparisons vary by model rather than randomness.
- Cross-validation score standard deviation: population standard deviation
  (`ddof=0`), matching NumPy's default for fold-score arrays. For five folds,
  sample standard deviation (`ddof=1`) is about 12% larger
  (`sqrt(5/4) ≈ 1.118`).
- Results schema: fixed columns include `n_features` so appended rows stay
  aligned with the header; create a new results file rather than changing the
  column layout.
  Absent vs truly missing: NA is filled with “None”/0 only when the house really lacks the feature (no garage, no basement...). If the feature exists but a value is missing, the pipeline imputes it.
LotFrontage: filled with the neighborhood median learned on the training fold, with the global median as a fallback.
Quality scales mapped to numbers (Po=1 ... Ex=5) because their order matters; nominal categories are one-hot encoded for linear models and integer-coded for trees.
MSSubClass and MoSold treated as categories; YrSold stays numeric.
is_remodeled ignores houses built before 1950 with a remodel year of 1950 (the column’s floor, not a real remodel).
Log1p only on skewed continuous columns (skew > 0.75, more than 10 distinct values), decided inside the pipeline.
Two preprocessors: linear (log + scaling + one-hot) and tree (no log, no scaling, integer codes).

Validation (Phase 3)

One fixed protocol: 5-fold KFold, shuffled, seed 42, defined once in src/cv.py. RMSE on log price, reported as mean ± std. The same folds for every model, so differences come from the model and models can be compared fold by fold.
Shuffling is fine here because the houses are independent. With time-ordered data I’d use TimeSeriesSplit instead.
RMSE is the main metric (it’s the competition metric). MAE and R² are reported alongside, because RMSE alone hides whether the error comes from many small misses or a few big ones.
Baselines to beat: 0.399 (mean), 0.269 (neighborhood), 0.231 (quality), 0.1995 (quality + log area).

Linear regression (Phase 4)

statsmodels for inference (coefficients, p-values, assumption tests), scikit-learn for CV scores. Same model, two jobs.
Breusch-Pagan rejects constant variance even with the log target, so p-values come from HC3 robust standard errors.
Plain OLS on all 280 features (0.1495) is worse than Ridge (0.1315): too many correlated columns without a penalty. Regularization comes next.
Remaining model issues: the extremes of quality and size are under-predicted, residuals cluster by neighborhood, and a few unusual sales (Partial, Abnorml) produce most of the large errors.

Regularization and outliers (Phase 5)

Regularization is the main improvement for linear models (OLS 0.1495 → Ridge/Lasso about 0.1315 on the same features). Ridge, Lasso and Elastic Net are equal within noise; Lasso uses only 86 features.
Tuning alpha on the same folds was optimistic by about 0.002–0.005 compared with nested CV, so differences of that size between tuned models aren’t meaningful.
Houses 524 and 1299 are removed from training only (partial sales of unfinished houses; better on 4 of 5 folds). They stay in every evaluation. The “remove everything above 4,000 sq ft” rule isn’t used, because 692 and 1183 are real sales.
Huber regression (keeping every house) does as well or better, and is the safer choice when the data can’t be cleaned by hand.

## Setup

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Notebook working directories can vary, so use `RAW_DIR` from `src.cv` instead
of relying on relative paths to the raw data:

```python
import pandas as pd
from src.cv import RAW_DIR

train = pd.read_csv(RAW_DIR / "train.csv")
```

## Data and results

Keep the original Kaggle files in `data/raw/` unchanged. Save any generated
cleaned data in `data/processed/`, plots in `reports/figures/`, and record each
model run as a separate row in `results.csv` with `src/cv.py`'s `log_result`
helper. The helper records the population standard deviation of fold scores
(`ddof=0`).
