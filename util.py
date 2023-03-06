import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from scipy import sparse


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class ExeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._stretltype = 1

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        encoder = LabelEncoder()
        n_samples, n_features = X.shape
        arr = np.zeros_like(X, dtype=np.int)

        for whlist in range(X.shape[1]):
            arr[:, whlist] = encoder.fit_transform(X[:, whlist])
        return arr
