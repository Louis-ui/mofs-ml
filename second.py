import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from dataClean import data_wash
from regression import regressionAndAnalysis
from DataPreprocessing import *

from sklearn.preprocessing import QuantileTransformer


# 读文件
file = pd.read_excel("database\OUT_GCMC_fur_tip5.xlsx")

# np禁止科学计数法显示
# np.set_printoptions(suppress=True,   precision=20, threshold=10,  linewidth=40)
# pd禁止科学计数法显示
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 去重
# file.drop_duplicates(inplace=True)
# file.reset_index(drop=True, inplace=True)

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
# labels = ["Hr_fur/tip5"]
labels = ['Henry_furfural', "Henry_ethanol", 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p',"Hr_fur/tip5", "Hr_eth/tip5"]

column = feature + labels

dataset = file[column]

# 去空
# dataset.dropna(inplace=True)

# 清洗
for item in labels:
    dataset = data_wash(dataset, item)

dataset_labels = dataset[labels]
dataset_select = dataset[feature]

dataset_labels['Henry_furfural'] = np.log(dataset_labels['Henry_furfural']+1)
dataset_labels['Henry_Tip5p'] = np.log(dataset_labels['Henry_Tip5p']+1)
dataset_labels['Henry_ethanol'] = np.log(dataset_labels['Henry_ethanol']+1)

dataset_labels['Hr_fur/tip5'] = np.log(dataset_labels['Hr_fur/tip5']+1)
# dataset_labels['Hr_eth/tip5'] = np.log(dataset_labels['Hr_eth/tip5']+1)
 
#移除低方差特征
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(dataset_select)

X_train, X_test, Y_train, Y_test = split(
    dataset_select, dataset_labels, test_size=0.2)

#归一化特征
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()
# scaler= std.fit(X_train)
# X_train=scaler.transform(X_train)
# X_test_scaler=scaler.transform(X_test)

# 数据预处理
X_train, X_test = imp(X_train, X_test)
X_train, X_test = scale(X_train, X_test)
Y_train, Y_test = normal_qt(Y_train, Y_test)
# Y_train, Y_test = scale(Y_train, Y_test)

# 训练分析
regressionAndAnalysis(labels, "model_RandomForestRegressor",
                      X_train, Y_train, X_test, Y_test)
