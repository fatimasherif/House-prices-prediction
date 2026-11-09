"""Simple reference models that every real model must beat."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict the median target of each group (e.g. each neighborhood).

    The medians are learned in fit(), so inside CV they come from the training
    fold only. Groups not seen in training get the overall median.
    """

    def __init__(self, group_col="Neighborhood"):
        self.group_col = group_col

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.medians_ = y.groupby(X[self.group_col]).median()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X):
        return X[self.group_col].map(self.medians_).fillna(self.global_median_).to_numpy(dtype=float)