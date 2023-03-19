import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mof import MOF
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
