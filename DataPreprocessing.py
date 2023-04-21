import numpy as np
import matplotlib.pyplot as plt
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


def quantileF(xtrain, xtest):
    scaler = QuantileTransformer(output_distribution='normal')
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)
    return xtrain, xtest


def preprocessing(dataset_select, dataset_labels, test_size=0.2):
    # 数据划分
    X_train, X_test, Y_train, Y_test = split(
        dataset_select, dataset_labels, test_size)

    # X_train, X_test = imp(X_train, X_test)
    X_train, X_test = scale(X_train, X_test)
    X_train, X_test = minScale(X_train, X_test)
    X_train, X_test = quantileF(X_train, X_test)
    # # Y_train, Y_test = normal_qt(Y_train, Y_test)

    # #绘制转换后的训练集数据的直方图
    # plt.subplot(2, 1, 1)
    # plt.hist(X_train, bins=30)
    # plt.title('训练集')

    # #增加子图之间的间隙
    # plt.subplots_adjust(hspace=0.5)
    # #显示图像
    # plt.show()
    # #绘制转换后的测试集数据的直方图
    # plt.subplot(2, 1, 1)
    # plt.hist(X_test, bins=25)
    # plt.title('测试集')

    # #增加子图之间的间隙
    # plt.subplots_adjust(hspace=0.5)
    # #显示图像
    # plt.show()

    return X_train, X_test, Y_train, Y_test

def pp(dataset_select, dataset_labels, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(dataset_select, dataset_labels, test_size=test_size, random_state=42)

    #正态化
    #创建QuantileTransformer对象，设置输出分布为正态分布
    quantile = QuantileTransformer(output_distribution='normal')
    #对训练集和测试集进行转换，并保存转换前的数据
    x_train_raw = x_train.copy()
    x_test_raw = x_test.copy()
    x_train = quantile.fit_transform(x_train)
    x_test = quantile.transform(x_test)
    #标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #找出离群值的索引，假设大于3或小于-3的值是离群值
    outlier_index_train = np.where(np.abs(x_train) > 3)
    outlier_index_test = np.where(np.abs(x_test) > 3)
    
    #删除原数据中对应的行
    x_train_raw = np.delete(x_train_raw, outlier_index_train[0], axis=0)
    y_train = np.delete(y_train, outlier_index_train[0], axis=0)
    x_test_raw = np.delete(x_test_raw, outlier_index_test[0], axis=0)
    y_test = np.delete(y_test, outlier_index_test[0], axis=0)
    x_train_raw  =  x_train_raw.reset_index(drop=True)
    #再次转换数据
    #标准化
    scaler = StandardScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    
    #正态化
    x_train_raw = quantile.fit_transform(x_train_raw)
    x_test_raw = quantile.transform(x_test_raw)
    quantile = QuantileTransformer(output_distribution='normal')
    y_train = quantile.fit_transform(y_train)
    y_test = quantile.transform(y_test)
    
    #归一化
    scaler = MinMaxScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)

    # #绘制转换后的训练集数据的直方图
    # plt.subplot(2, 1, 1)
    # plt.hist(x_train_raw, bins=30)
    # plt.title('训练集')

    # #增加子图之间的间隙
    # plt.subplots_adjust(hspace=0.5)
    # #显示图像
    # plt.show()
    # #绘制转换后的测试集数据的直方图
    # plt.subplot(2, 1, 1)
    # plt.hist(x_test_raw, bins=25)
    # plt.title('测试集')

    # #增加子图之间的间隙
    # plt.subplots_adjust(hspace=0.5)
    # #显示图像
    # plt.show()


    return x_train_raw, x_test_raw, y_train, y_test
