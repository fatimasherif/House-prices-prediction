# House Prices: Advanced Regression Techniques

An end-to-end regression project on the Kaggle House Prices dataset: exploration, cleaning and feature engineering, a fixed validation protocol, linear models with full diagnostics, regularization, tree models and boosting, tuning, interpretation, blending and a Kaggle submission.

The goal was to understand regression in depth rather than to chase the leaderboard, so every step is measured with the same cross-validation and every decision is written down below.

## Results

| Step | CV RMSE (log price) |
|---|---|
| Predict the mean | 0.399 |
| OverallQual + log(GrLivArea), linear regression | 0.1995 |
| All features, plain OLS | 0.1495 |
| Ridge / Lasso | 0.1315 |
| Ridge, houses 524 and 1299 out of training | 0.1277 |
| Best single model (CatBoost) | 0.1219 |
| **Final blend** (equal-weight average of 6 models) | **0.1193 ± 0.021** |

**Kaggle public score: 0.12081**, close to the CV estimate (0.1193), so the validation protocol was a reliable guide.

The final blend averages Ridge, Huber, SVR, LightGBM, XGBoost and CatBoost with equal weights. RMSE on log price reads roughly as a percentage error, so the final model is off by about 12% on a typical house.

## Project layout

```text
data/
  raw/                  Original Kaggle files (not in git; never edited)
  processed/
    oof_predictions.csv Out-of-fold predictions of every model, used for blending
notebooks/
  01_eda.ipynb                    Target, strongest predictors, outliers, missing values
  02_cleaning_fe.ipynb            Cleaning, feature engineering, preprocessing pipelines
  03_validation_baselines.ipynb   The validation protocol and four baselines
  04_linear_diagnostics.ipynb     OLS with statsmodels: coefficients, assumptions, VIF, influence
  05_regularization.ipynb         Ridge, Lasso, Elastic Net, nested CV, the outlier experiment
  06_nonlinear_models.ipynb       Polynomial, KNN, SVR, trees, random forest, boosting
  07_tuning_interpretation.ipynb  Optuna, learning curves, permutation importance, SHAP
  08_blending_submission.ipynb    Blending, error analysis, the Kaggle submission
src/
  cv.py          Seed, paths, the shared folds, evaluation and results logging
  features.py    clean(), the custom transformers and the preprocessing pipelines
  baseline.py    Neighborhood-median baseline model
  models.py      CatBoost wrapper that works inside scikit-learn pipelines
reports/
  figures/       Saved plots, prefixed with the phase number
submissions/     Kaggle submission files, named with the date and CV score
results.csv      One row per model run, same folds and metric for all
NOTES.md         Working notes (expected / got / because)
requirements.txt Exact package versions
```

## How to run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name house-prices --display-name "Python (house-prices)"
```

Put `train.csv`, `test.csv` and `data_description.txt` from Kaggle in `data/raw/`, then run the notebooks in order with the "Python (house-prices)" kernel. Notebook 08 writes the submission file to `submissions/`.

Notebooks load data through `RAW_DIR` from `src/cv.py`, so they don't depend on the working directory:

```python
import pandas as pd
from src.cv import RAW_DIR

train = pd.read_csv(RAW_DIR / "train.csv")
```

## Decision log

### Setup (Phase 0)
- Python 3.12 in a project venv: every library in the project has stable builds for it, and the project doesn't depend on the Anaconda base environment.
- One seed for everything: `SEED = 42`, passed as `random_state` to every splitter and model, so results are reproducible and comparisons differ only by the model.
- The fold standard deviation uses `ddof=0` (population std), matching NumPy's default. With 5 folds, `ddof=1` would be about 12% larger (√(5/4) ≈ 1.118).
- `results.csv` has a fixed set of columns. Re-running the same model on the same feature set replaces its row, so re-running a notebook never creates duplicates.

### Target (Phase 1)
- The models predict `log1p(SalePrice)`. The raw price is right-skewed (skew 1.88, 0.12 after the log), the log makes errors relative instead of dollar-based, and it matches the competition metric. Predictions go back to dollars with `expm1`.

### Cleaning and features (Phase 2)
- Absent vs truly missing: NA is filled with "None"/0 only when the house really lacks the feature (no garage, no basement...). If the feature exists but a value is missing, the pipeline imputes it.
- LotFrontage is filled with the neighborhood median learned on the training fold, with the global median as a fallback.
- Quality scales are mapped to numbers (Po=1 ... Ex=5) because their order matters. Nominal categories are one-hot encoded for linear models and integer-coded for trees; CatBoost gets them as text.
- MSSubClass and MoSold are treated as categories; YrSold stays numeric.
- `is_remodeled` ignores houses built before 1950 with a remodel year of 1950 (the column's floor, not a real remodel). Without this fix, 166 houses were wrongly counted as remodeled and the feature's correlation had the wrong sign.
- log1p only on skewed continuous columns (skew > 0.75, more than 10 distinct values), decided inside the pipeline.
- Everything that learns from data (medians, skew, scaling, encoding) sits inside a scikit-learn Pipeline, so in CV it only ever sees the training fold.
- Two preprocessors: linear (log + scaling + one-hot) and tree (no log, no scaling, integer codes).

### Validation (Phase 3)
- One fixed protocol: 5-fold KFold, shuffled, seed 42, defined once in `src/cv.py`. RMSE on log price, reported as mean ± std. The same folds for every model, so differences come from the model and models can be compared fold by fold.
- Shuffling is fine here because the houses are independent. With time-ordered data I'd use TimeSeriesSplit instead.
- RMSE is the main metric (it's the competition metric). MAE and R² are reported alongside, because RMSE alone hides whether the error comes from many small misses or a few big ones.
- Baselines to beat: 0.399 (mean), 0.269 (neighborhood), 0.231 (quality), 0.1995 (quality + log area).

### Linear regression (Phase 4)
- statsmodels for inference (coefficients, p-values, assumption tests), scikit-learn for CV scores. Same model, two jobs.
- Breusch-Pagan rejects constant variance even with the log target, so p-values come from HC3 robust standard errors.
- Plain OLS on all 280 features (0.1495) is worse than Ridge (0.1315): too many correlated columns without a penalty.
- Remaining model issues: the extremes of quality and size are under-predicted, residuals cluster by neighborhood, and a few unusual sales (Partial, Abnorml) produce most of the large errors.

### Regularization and outliers (Phase 5)
- Regularization is the main improvement for linear models (OLS 0.1495 → Ridge/Lasso about 0.1315 on the same features). Ridge, Lasso and Elastic Net are equal within noise; Lasso uses only 86 features.
- Tuning alpha on the same folds was optimistic by about 0.002–0.005 compared with nested CV, so differences of that size between tuned models aren't meaningful.
- Houses 524 and 1299 are removed from training only (partial sales of unfinished houses; better on 4 of 5 folds). They stay in every evaluation. The "remove everything above 4,000 sq ft" rule isn't used, because 692 and 1183 are real sales.
- Huber regression (keeping every house) does as well or better, and is the safer choice when the data can't be cleaned by hand.

### Non-linear models (Phase 6)
- Boosting is the best family (CatBoost and XGBoost about 0.122, untuned), but only slightly better than Ridge (0.128) and on 3 of 5 folds, much of it from the fold with the two outlier houses. Small data and near-linear relationships in log space keep linear models competitive.
- Polynomial terms, KNN and a single tree don't help. Random forest is worse than Ridge here.
- Trees are less sensitive to outliers than linear models, but they can't extrapolate beyond the largest house in training.
- Linear and tree models make different errors (correlation about 0.85), so blending them is worth it.

### Tuning and interpretation (Phase 7)
- Optuna with TPE beat random search with the same budget (0.1231 vs 0.1244). Pruning stopped bad trials early.
- The honest (nested CV) gain from tuning LightGBM is about 0.001 (0.1242 vs 0.1252). Features and outlier handling mattered far more than hyperparameters.
- Learning curves: Ridge is limited by bias, LightGBM by variance. On normal houses they're equal.
- All interpretation tools (Lasso coefficients, split gain, permutation importance, SHAP) agree that quality and size drive the price. Where they disagree, it's because of raw vs engineered features, unscaled one-hot coefficients and integer-coded neighborhoods.

### Blending and submission (Phase 8)
- The final model is an equal-weight average of Ridge, Huber, SVR, LightGBM, XGBoost and CatBoost (CV 0.1193). Every blend beat the best single model on 4 of 5 folds; mixing linear and tree models gave the gain.
- Learned weights (honest 0.1202) and stacking (0.1215) did no better than a plain average: with similar models and little data, learned combinations overfit.
- Unusual sales (Partial, Abnorml, Family, Alloca) are 18% of the houses but about 55% of the squared error. The cheapest and most expensive houses are also harder than the middle.
- Kaggle public score 0.12081 vs CV 0.1193: the gap is far smaller than the spread between folds, so CV was a reliable estimate.

## Next steps
- Target encoding for Neighborhood (the trees only get it as an integer code, and location errors remain).
- A sale-condition feature or a separate treatment for non-normal sales, which cause most of the remaining error.