import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

# 特征缩放  标准化


def scale(Xtrain, Xtest):
    scaler = StandardScaler().fit(Xtrain)
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

def minScale(xtrain, xtest):
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)
    return xtrain, xtest

def quantile(xtrain, xtest):
    scaler =  QuantileTransformer(output_distribution='normal')
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)
    return xtrain, xtest

def preprocessing(dataset_select, dataset_labels, test_size=0.2):
    # 数据划分
    X_train, X_test, Y_train, Y_test = split(
        dataset_select, dataset_labels, test_size)

    x_train_raw = X_train.copy()
    x_test_raw = X_test.copy()

    # 数据预处理
    X_train, X_test = imp(X_train, X_test)
    X_train, X_test = scale(X_train, X_test)
    X_train, X_test = quantile(X_train, X_test)
    X_train, X_test = minScale(X_train, X_test)

    # # 找出离群值的索引，假设大于3或小于-3的值是离群值
    outlier_index_train = np.where(np.abs(X_train) > 3)
    outlier_index_test = np.where(np.abs(X_test) > 3)
    print("Train Outlier Index:", outlier_index_train)
    print("Test Outlier Index:", outlier_index_test)

    # # 删除原数据中对应的行
    x_train_raw = np.delete(x_train_raw, outlier_index_train[0], axis=0)
    Y_train = np.delete(Y_train, outlier_index_train[0], axis=0)
    x_test_raw = np.delete(x_test_raw, outlier_index_test[0], axis=0)
    Y_test = np.delete(Y_test, outlier_index_test[0], axis=0)
    x_train_raw = x_train_raw.reset_index(drop=True)
    x_test_raw = x_test_raw.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    X_train, X_test = imp(x_train_raw, x_test_raw)
    X_train, X_test = scale(x_train_raw, x_test_raw)
    X_train, X_test = minScale(X_train, X_test)

    return X_train, X_test, Y_train, Y_test
