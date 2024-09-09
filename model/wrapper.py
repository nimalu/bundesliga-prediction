import numpy as np

from sklearn.base import BaseEstimator


class RoundingWrapper(BaseEstimator):
    def __init__(self, regressor) -> None:
        self.regressor = regressor
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
    
    def predict(self, X):
        raw = self.regressor.predict(X)
        return np.round(raw)

class DifferenceWrapper(BaseEstimator):
    def __init__(self, regressor, base=0, **kwargs) -> None:
        self.regressor = regressor
        self.regressor.set_params(**kwargs)
        self.base = base

    def fit(self, X, y):
        y_diff = y[:, 0] - y[:, 1]
        self.min_diff = np.min(y_diff)
        self.regressor.fit(X, y_diff - self.min_diff)
    
    def predict(self, X):
        y_diff = self.regressor.predict(X) + self.min_diff
        results = np.zeros((len(X), 2))
        results[:, 1] = self.base
        results[:, 0] = results[:, 1] + y_diff
        return np.round(np.clip(results, 0, 10))
