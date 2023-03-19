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

# 读文件
file = pd.read_excel("database\data.xlsx")

# np禁止科学计数法显示
np.set_printoptions(suppress=True,   precision=20, threshold=10,  linewidth=40)
# pd禁止科学计数法显示
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 去重
file.drop_duplicates(inplace=True)
file.reset_index(drop=True, inplace=True)

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
labels = ['Henry_furfural','Henry_Tip5p','Heat_furfural','Heat_Tip5p']

dataset = file[feature + labels]
dataset.dropna(inplace=True)

# 清除LCD小于糠醛分子的动力学直径（5.7）的MOFS
MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL = 5.7
dataset_drop_small_LCD = dataset[dataset['LCD']>=MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL]

dataset_labels = dataset_drop_small_LCD[labels]
dataset_select = dataset_drop_small_LCD[feature]

# dataset_labels = dataset[labels]
# dataset_select = dataset[feature]

dataset_labels['Henry_furfural'] = np.log(dataset_labels['Henry_furfural'])
dataset_labels['Henry_Tip5p'] = np.log(dataset_labels['Henry_Tip5p'])

dataset_labels['selectivity_of_Henry'] = dataset_labels.apply(lambda x: x["Henry_furfural"] / x["Henry_Tip5p"], axis=1)
dataset_labels['selectivity_of_Heat'] = dataset_labels.apply(lambda x: x["Heat_furfural"] / x["Heat_Tip5p"], axis=1)
labels.append('selectivity_of_Henry')
labels.append('selectivity_of_Heat')

# 对每一个标签都进行回归
for item in labels:
    #真正需要回归的目标
    target_label = [item]
    dataset_target_labels = dataset_labels[target_label]
    # 初始化数据
    Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare = initData(dataset_select, dataset_target_labels, feature, target_label)
    # 开始训练
    regressionAndAnalysis(target_label, "model_RandomForestRegressor" , Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare)
    print("------分割线------")


# target_label = ['Henry_furfural']
# dataset_target_labels = dataset_labels[target_label]

