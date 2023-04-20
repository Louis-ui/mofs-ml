import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataClean import data_wash
from regression import regressionAndAnalysis
from DataPreprocessing import *

warnings.filterwarnings('ignore')

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
# labels = ['Henry_furfural', 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p']
labels = ['Henry_furfural']

column = feature + labels

dataset = file[column]

# labels += ['selectivity_of_Henry']

# column += ['selectivity_of_Henry']

# dataset['selectivity_of_Henry'] = dataset['Henry_furfural'] / dataset['Henry_Tip5p']

# 去空
# dataset.dropna(inplace=True)

# 清除LCD小于糠醛分子的动力学直径（5.7）的MOFS  --  有问题，应该去除苯环
# MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL = 5.7
# dataset_drop_small_LCD = dataset[dataset['LCD']>=MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL]
# dataset_labels = dataset_drop_small_LCD[labels]
# dataset_select = dataset_drop_small_LCD[feature]

# 清洗
for item in labels:
    dataset = data_wash(dataset, item)

dataset_select = dataset[feature]
dataset_labels = dataset[labels]

dataset_labels['Henry_furfural'] = np.log(dataset_labels['Henry_furfural'] + 1)
# dataset_labels['Henry_Tip5p'] = np.log(dataset_labels['Henry_Tip5p']+1)
# dataset_labels['selectivity_of_Henry'] = np.log(
#     dataset_labels['selectivity_of_Henry']+1)

#移除低方差特征
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(dataset_select)

# 数据划分
X_train, X_test, Y_train, Y_test = split(dataset_select, dataset_labels, test_size=0.2)

# 数据预处理
X_train , X_test = imp(X_train, X_test)
X_train , X_test = scale(X_train, X_test)
Y_train, Y_test = normal_qt(Y_train, Y_test)
# Y_train, Y_test = scale(Y_train, Y_test)

# 训练分析
regressionAndAnalysis(labels, "model_RandomForestRegressor" , X_train, Y_train, X_test, Y_test)
