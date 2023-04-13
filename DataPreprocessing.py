import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# 数据划分
def split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test


# 插补特征缺失值
def imp(Xtrain, Xtest):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(Xtrain)
    Xtrain_imp = imp.transform(Xtrain)
    Xtest_imp = imp.transform(Xtest)
    return Xtrain_imp, Xtest_imp

# 特征缩放
def scale(Xtrain, Xtest):
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    X_train_std = scaler.transform(Xtrain)
    X_test_std = scaler.transform(Xtest)
    return X_train_std, X_test_std

# 目标值缩放
def normal_qt(ytrain, ytest):
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    qt.fit(ytrain)
    y_train_normal = qt.transform(ytrain)
    y_test_normal = qt.transform(ytest)
    return y_train_normal, y_test_normal


def normal_pt(ytrain, ytest):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    pt.fit(ytrain)
    y_train_normal = pt.transform(ytrain)
    y_test_normal = pt.transform(ytest)
    return y_train_normal, y_test_normal


# 目标值反缩放
def normal_qt_inv(ytrain, ytest):
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    qt.fit(ytrain)
    y_train_normal = qt.inverse_transform(ytrain)
    y_test_normal = qt.inverse_transform(ytest)
    return y_train_normal, y_test_normal


def normal_pt_inv(ytrain, ytest):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    pt.fit(ytrain)
    y_train_normal = pt.inverse_transform(ytrain)
    y_test_normal = pt.inverse_transform(ytest)
    return y_train_normal, y_test_normal
