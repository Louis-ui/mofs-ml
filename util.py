import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt


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


# 可视化函数
def plot_hist(data, column):
    plt.figure(figsize=(10, 10))
    for i in column:
        plt.subplot(3, 3, column.index(i)+1)
        data[i].plot(kind='hist', bins=50, grid=True)
        plt.title(i)
    plt.show()


def whatData(features, label, dataset):
    fig,  axes = plt.subplots(
        nrows=1,  ncols=len(features),  figsize=(200,  5))
    for i,  ax in enumerate(axes):
        sns.scatterplot(x=dataset[features[i]],
                        y=dataset[label],    alpha=0.6,  ax=ax)
        ax.set_xlabel(features[i],    fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title('特征关系:' + features[i] + '\n'
                     '和' + label, fontsize=14)
    plt.tight_layout()
    plt.show()
