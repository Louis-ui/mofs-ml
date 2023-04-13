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
from dataClean import initData
from regression import regressionAndAnalysis
# from chemicalDescriptor import chemicDescriptorData_prepare

from sklearn.preprocessing import QuantileTransformer


# 读文件
file = pd.read_excel("database\OUT_GCMC_fur_tip5.xlsx")

# np禁止科学计数法显示
np.set_printoptions(suppress=True,   precision=20, threshold=10,  linewidth=40)
# pd禁止科学计数法显示
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 去重
file.drop_duplicates(inplace=True)
file.reset_index(drop=True, inplace=True)

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g', 'Henry_furfural','Henry_Tip5p', ]
labels = ['Heat_furfural','Heat_Tip5p',"Hr_fur/tip5","Hr_eth/tip5"]

dataset = file[feature + labels]
dataset.dropna(inplace=True)

dataset_labels = dataset[labels]
dataset_select = dataset[feature]

dataset_select['Henry_furfural'] = np.log(dataset_select['Henry_furfural'])
dataset_select['Henry_Tip5p'] = np.log(dataset_select['Henry_Tip5p'])

dataset_labels['Hr_fur/tip5'] = np.log(dataset_labels['Hr_fur/tip5'])
dataset_labels['Hr_eth/tip5'] = np.log(dataset_labels['Hr_eth/tip5'])


# # 对每一个标签都进行回归
for item in labels:
    #真正需要回归的目标
    target_label = [item]
    dataset_target_labels = dataset_labels[target_label]
    # 初始化数据
    Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare = initData(dataset_select, dataset_target_labels, feature, target_label)
    # 正态分布
    Xtrain_prepare = QuantileTransformer(random_state=1000,n_quantiles=len(Xtrain_prepare)).fit_transform(Xtrain_prepare)
    Ytrain_prepare = QuantileTransformer(random_state=1000,n_quantiles=len(Ytrain_prepare)).fit_transform(Ytrain_prepare)
    Xtest_prepare = QuantileTransformer(random_state=1000,n_quantiles=len(Xtest_prepare)).fit_transform(Xtest_prepare)
    Ytest_prepare = QuantileTransformer(random_state=1000,n_quantiles=len(Ytest_prepare)).fit_transform(Ytest_prepare)

    # 开始训练
    regressionAndAnalysis(target_label, "model_RandomForestRegressor" , Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare)
    print("------分割线------")