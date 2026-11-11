"""Model wrappers that need a little extra glue to work inside scikit-learn pipelines."""

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostNativeCats(RegressorMixin, BaseEstimator):
    """CatBoost that finds the text (categorical) columns itself when it is fitted.

    Passing cat_features to CatBoostRegressor directly breaks scikit-learn's clone(),
    which CV needs; this wrapper keeps every constructor argument simple.
    """

    def __init__(self, iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                 random_seed=42, thread_count=-1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.cat_features_ = X.select_dtypes(exclude="number").columns.tolist()
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_seed,
            thread_count=self.thread_count, cat_features=self.cat_features_, verbose=0,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)