import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from matplotlib import rcParams
from pylab import mpl

warnings.filterwarnings('ignore')

# 读文件
file = pd.read_excel("database\OUT_GCMC_fur_tip5.xlsx")

feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
labels = ['P1e+07_Con0_mg/g', 'P1e+07_Con1_mg/g', 'P1e+07_S_mg/g']

column = feature + labels

dataset = file[column]

def data_wash_replace(data, column):
    # 去重
    data = data.drop_duplicates()
    # 目标值为None

    def del_nan(data, column):
        data_del = data.dropna(axis=0, how='any', subset=column)
        return data_del

    def replace(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        max_value = Q3 + 1.5*IQR
        min_value = Q1 - 1.5*IQR
        outliers = data[(data[column] > max_value) |
                        (data[column] < min_value)].index
        median = data[column].median()
        data.loc[outliers,  column] = median
        return data

    data = del_nan(data, column)
    data[column] = data[column].replace(0, np.nan)
    data[column] = data[column].fillna(data[column].mean()) 
    data = del_nan(data, column)
    # data = replace(data, column)

    return data

# 清洗
for item in column:
    dataset = data_wash_replace(dataset, item)


def preprocessing(dataset_select, dataset_labels, test_size=0.2):
    # 数据划分
    def split(X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test
    X_train, X_test, Y_train, Y_test = split(
        dataset_select, dataset_labels, test_size)
    
    def scale(Xtrain, Xtest):
        scaler = StandardScaler().fit(Xtrain)
        X_train_std = scaler.transform(Xtrain)
        X_test_std = scaler.transform(Xtest)
        return X_train_std, X_test_std
    
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

    X_train, X_test = scale(X_train, X_test)
    X_train, X_test = minScale(X_train, X_test)
    # X_train, X_test = quantileF(X_train, X_test)

    Y_train, Y_test = scale(Y_train, Y_test)
    Y_train, Y_test = minScale(Y_train, Y_test)
    # Y_train, Y_test = quantileF(Y_train, Y_test)

    return X_train, X_test, Y_train, Y_test

def regressionAnalysis(method, target, y_true, y_pred):
    print('%s训练的结果' % target[0])
    print('使用%s训练' % method)
    RMSE = mean_squared_error(y_true,y_pred)
    print('均方根误差RMSE: %f' % RMSE)
    MSE = mean_squared_error(y_true,y_pred)
    print('均方误差MSE: %f' % MSE)
    MAE = mean_absolute_error(y_true,y_pred)
    print('平均绝对误差MAE: %f' % MAE)
    R2 = r2_score(y_true,y_pred)
    print('R2: %f' % R2)

models = {
    "model_LinearRegression": linear_model.LinearRegression(),
    "model_DecisionTreeRegressor": tree.DecisionTreeRegressor(),
    "model_SVR": svm.SVR(),
    "model_KNeighborsRegressor": neighbors.KNeighborsRegressor(),
    "model_RandomForestRegressor": ensemble.RandomForestRegressor(n_estimators=20),
    "model_AdaBoostRegressor": ensemble.AdaBoostRegressor(n_estimators=50),
    "model_GradientBoostingRegressor": ensemble.GradientBoostingRegressor(n_estimators=100),
    "model_BaggingRegressor": BaggingRegressor(),
    "model_ExtraTreeRegressor":  ExtraTreeRegressor()
}

def singleRAWithDiffModel(target_labels, Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare, X_data):
    for function in models:
        model = models[function]
        model.fit(Xtrain_prepare, Ytrain_prepare)
        predictions = model.predict(Xtest_prepare)
        for index in range(Ytest_prepare.shape[1]):
            target_label = [target_labels[index]]
            regressionAnalysis(function, target_label, Ytest_prepare, predictions)
            # 特征重要程度分析
            # important(X_data, model)

def singleRA(target_labels, method, Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare, X_data):
    model = models[method]
    model.fit(Xtrain_prepare, Ytrain_prepare)
    predictions = model.predict(Xtest_prepare)
    for index in range(Ytest_prepare.shape[1]):
        target_label = [target_labels[index]]
        regressionAnalysis(method, target_label, Ytest_prepare, predictions)
        # 特征重要程度分析
        # important(X_data, model)
        # importantWithShape(x_data=X_data, model=model)

# 真正想训练的内容
for i in labels:
    true_label = [i]
    dataset_select = dataset[feature]
    dataset_labels = dataset[true_label]
    X_train, X_test, Y_train, Y_test = preprocessing(
        dataset_select, dataset_labels, test_size=0.2)
    singleRA(true_label, "model_RandomForestRegressor",
             X_train, Y_train, X_test, Y_test, dataset_select)
    # singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)

# # 真正想训练的内容
# true_label = ['P1e+07_Con0_mg/g']
# dataset_select = dataset[feature]
# dataset_labels = dataset[true_label]
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
# singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)
# print('----------------------------------  ')
# true_label = ['P1e+07_Con1_mg/g']
# dataset_select = dataset[feature]
# dataset_labels = dataset[true_label]
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
# singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)
# print('----------------------------------  ')
# true_label = ['P1e+07_S_mg/g']
# dataset_select = dataset[feature]
# dataset_labels = dataset[true_label]
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
# singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)
# print('----------------------------------  ')

