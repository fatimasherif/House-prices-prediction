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

- Python version: 3.14.5
- Shared random seed: `SEED = 42`
- Cross-validation score standard deviation: population standard deviation
  (`ddof=0`)

## Setup

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start Jupyter from the project root to use the relative data paths in the
notebooks.

## Data and results

Keep the original Kaggle files in `data/raw/` unchanged. Save any generated
cleaned data in `data/processed/`, plots in `reports/figures/`, and record each
model run as a separate row in `results.csv` with `src/cv.py`'s `log_result`
helper. The helper records the population standard deviation of fold scores
(`ddof=0`).
