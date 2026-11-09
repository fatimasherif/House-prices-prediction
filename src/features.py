"""Cleaning and feature engineering shared by train, CV folds and test.

Everything in clean() works row by row and learns nothing from the data
(no medians, no counts, no encodings), so it gives the same result whether
it runs on the full train set, on one CV fold or on test. Anything that
learns from data lives in the Pipeline instead.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# Ordinal scales: the order carries meaning, so we map them to numbers
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
OTHER_ORDINALS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
}

# Groups of columns that describe one feature of the house.
# NA in these columns means "absent" only when the house really lacks the feature.
BSMT_CATS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
BSMT_NUMS = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
GARAGE_CATS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
GARAGE_NUMS = ["GarageCars", "GarageArea"]

# No numeric partner: NA can only mean "none"
ALWAYS_NONE = ["Alley", "Fence", "MiscFeature"]

PORCH_COLS = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]


def _fill_absent(df, cat_cols, num_cols, has_feature):
    """Fill NA with "None" / 0 only for houses that don't have the feature.

    Houses that HAVE the feature but a missing value keep their NaN,
    so the Pipeline imputes them later (truly missing values).
    """
    absent = ~has_feature
    for col in cat_cols:
        df.loc[absent & df[col].isna(), col] = "None"
    for col in num_cols:
        df.loc[absent & df[col].isna(), col] = 0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with fixed types, absent values filled and new features."""
    df = df.copy()

    # 1. Flags: does the house have this feature at all?
    has_basement = df["TotalBsmtSF"].fillna(0) > 0
    has_garage = df["GarageType"].notna()   # the garage type is the most reliable signal
    has_pool = df["PoolArea"] > 0
    has_fireplace = df["Fireplaces"] > 0
    has_veneer = df["MasVnrArea"].fillna(0) > 0

    # 2. "Absent" NAs: fill only where the feature really doesn't exist
    _fill_absent(df, BSMT_CATS, BSMT_NUMS, has_basement)
    _fill_absent(df, GARAGE_CATS, GARAGE_NUMS, has_garage)
    _fill_absent(df, ["PoolQC"], [], has_pool)
    _fill_absent(df, ["FireplaceQu"], [], has_fireplace)
    _fill_absent(df, ["MasVnrType"], ["MasVnrArea"], has_veneer)
    for col in ALWAYS_NONE:
        df[col] = df[col].fillna("None")

    # 3. Garage year: one test house has 2207 (a typo); no garage -> use the house's year
    df.loc[df["GarageYrBlt"] > 2010, "GarageYrBlt"] = np.nan
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # 4. Codes that look like numbers but are categories
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)

    # 5. Ordinal scales -> numbers (unknown or truly missing values stay NaN)
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP).astype(float)
    for col, mapping in OTHER_ORDINALS.items():
        df[col] = df[col].map(mapping).astype(float)

    # 6. Has-flags as 0/1 features: absence itself carries price information
    df["has_basement"] = has_basement.astype(int)
    df["has_garage"] = has_garage.astype(int)
    df["has_pool"] = has_pool.astype(int)
    df["has_fireplace"] = has_fireplace.astype(int)
    df["has_2nd_floor"] = (df["2ndFlrSF"] > 0).astype(int)

    # 7. Engineered features, each one tests a hypothesis
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    # Ages can't be negative (a few houses were sold before they were finished)
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    # YearRemodAdd never goes below 1950 in this data: older houses that were
    # never remodeled show 1950, so they must not count as remodeled
    floor_artifact = (df["YearRemodAdd"] == 1950) & (df["YearBuilt"] < 1950)
    df["is_remodeled"] = ((df["YearRemodAdd"] != df["YearBuilt"]) & ~floor_artifact).astype(int)
    df["TotalPorchSF"] = df[PORCH_COLS].sum(axis=1)
    df["QualxArea"] = df["OverallQual"] * df["GrLivArea"]

    return df


# ---------------------------------------------------------------------------
# Transformers that LEARN from data: they only see the training fold in fit()
# ---------------------------------------------------------------------------

class LotFrontageImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Fill missing LotFrontage with the median of the house's neighborhood.

    The medians are learned in fit(), so inside CV they come from the training
    fold only. Neighborhoods with no known value (or unseen ones) fall back to
    the global median.
    """

    def __init__(self, group_col="Neighborhood", target_col="LotFrontage"):
        self.group_col = group_col
        self.target_col = target_col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.group_medians_ = X.groupby(self.group_col)[self.target_col].median()
        self.global_median_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group_col].map(self.group_medians_).fillna(self.global_median_)
        X[self.target_col] = X[self.target_col].fillna(fill)
        return X


class SkewedLog1p(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply log1p to right-skewed, non-negative, continuous columns.

    Which columns count as skewed is decided in fit(), on the training fold only.
    Columns with few distinct values (ordinal scales, 0/1 flags, small counts)
    are left alone: logging them would only distort their equal steps.
    """

    def __init__(self, threshold=0.75, min_unique=10):
        self.threshold = threshold
        self.min_unique = min_unique

    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X, dtype=float))
        self.n_features_in_ = X.shape[1]
        skewed = (
            (X.skew() > self.threshold)
            & (X.min() >= 0)
            & (X.nunique() > self.min_unique)
        )
        self.columns_ = np.flatnonzero(skewed.to_numpy())
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.columns_] = np.log1p(X[:, self.columns_])
        return X


def make_linear_preprocessor():
    """Raw data -> clean -> impute -> log skewed -> scale numerics, one-hot categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log_skewed", SkewedLog1p(threshold=0.75)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean)),
        ("lot_frontage", LotFrontageImputer()),
        ("columns", columns),
    ])


def make_tree_preprocessor():
    """Raw data -> clean -> impute; no log, no scaling; categories as integer codes."""
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    columns = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("clean", FunctionTransformer(clean)),
        ("lot_frontage", LotFrontageImputer()),
        ("columns", columns),
    ])


# ---------------------------------------------------------------------------
# Small, readable feature set for OLS inference (Phase 4)
# ---------------------------------------------------------------------------

def make_ols_subset(df: pd.DataFrame) -> pd.DataFrame:
    """17 hand-picked features with logs on the skewed areas.

    Small enough to read every coefficient and p-value. Works on raw data
    (it calls clean() first), so it can sit at the start of a Pipeline.
    """
    c = clean(df)
    out = pd.DataFrame(index=c.index)
    out["OverallQual"] = c["OverallQual"]
    out["OverallCond"] = c["OverallCond"]
    out["log_GrLivArea"] = np.log(c["GrLivArea"])
    out["log1p_TotalBsmtSF"] = np.log1p(c["TotalBsmtSF"])
    out["log_LotArea"] = np.log(c["LotArea"])
    out["GarageCars"] = c["GarageCars"]
    out["GarageArea"] = c["GarageArea"]
    out["YearBuilt"] = c["YearBuilt"]
    out["YearRemodAdd"] = c["YearRemodAdd"]
    out["FullBath"] = c["FullBath"]
    out["HalfBath"] = c["HalfBath"]
    out["TotRmsAbvGrd"] = c["TotRmsAbvGrd"]
    out["Fireplaces"] = c["Fireplaces"]
    out["KitchenQual"] = c["KitchenQual"]
    out["ExterQual"] = c["ExterQual"]
    out["BsmtQual"] = c["BsmtQual"]
    out["CentralAir_Y"] = (c["CentralAir"] == "Y").astype(int)
    return out