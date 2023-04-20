import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bean.mof import MOF
from normalization import max_min_normalization
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from util import DataFrameSelector


def initData(dataset_select, dataset_target_labels, feature, target_label):
    # 训练集测试集划分
    random_state = 42
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        dataset_select, dataset_target_labels, test_size=0.25, random_state=random_state)
    # 流水线清理数据
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(feature)),
        ('simple_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

    num_label_pipeline = Pipeline([
        ('selector', DataFrameSelector(target_label)),
        ('simple_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline)
    ])

    full_label_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_label_pipeline)
    ])

    Xtrain_prepare = full_pipeline.fit_transform(Xtrain)
    Xtest_prepare = full_pipeline.fit_transform(Xtest)

    Ytrain_prepare = full_label_pipeline.fit_transform(Ytrain)
    Ytest_prepare = full_label_pipeline.fit_transform(Ytest)

    return Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare


def data_wash_delete(data, column):
    # 清除LCD小于糠醛分子的动力学直径（5.7）的MOFS  --  有问题，应该去除苯环
    # MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL = 5.7
    # data = data[data['LCD']>=MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL]

    # 换无限为Nan
    data = data.replace([np.inf, -np.inf], np.nan)
    # 去重
    data = data.drop_duplicates()
    # 目标值为0
    def del_zero(data, column):
        querySe = data[(data[column] == 0)].index.tolist()
        data_del = data.drop(querySe, axis=0)
        return data_del
    # 目标值为None
    def del_nan(data, column):
        data_del = data.dropna(axis=0, how='any', subset=column)
        return data_del
    # 大和小
    def remove_outlier(data, column):
        Q1 = data[column].quantile(0.035)
        Q3 = data[column].quantile(0.965)
        IQR = Q3 - Q1
        data = data[(data[column] >= Q1 - 1.5 * IQR) &
                    (data[column] <= Q3 + 1.5 * IQR)]
        return data
    
    data[column] = data[column].replace(0, np.nan)
    data[column] = data[column].fillna(data[column].mean())
    data = data.dropna(axis=0, how='any')
    data = del_zero(data, column)
    data = del_nan(data, column)
    data = remove_outlier(data, column)
    return data


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

    data[column] = data[column].replace(0, np.nan)
    data[column] = data[column].fillna(data[column].mean()) 
    data = data.dropna(axis=0, how='any')
    data = del_nan(data, column)
    data = replace(data, column)

    return data

def data_clean_delete(data, column):
     # 换无限为Nan
    data = data.replace([np.inf, -np.inf], np.nan)
    # 去重
    data = data.drop_duplicates()
    # 目标值为0
    def del_zero(data, column):
        querySe = data[(data[column] == 0)].index.tolist()
        data_del = data.drop(querySe, axis=0)
        return data_del
    # 目标值为None
    def del_nan(data, column):
        data_del = data.dropna(axis=0, how='any', subset=column)
        return data_del
    data = del_zero(data, column)
    data = del_nan(data, column)
    data = data.dropna(axis=0, how='any')
    return data