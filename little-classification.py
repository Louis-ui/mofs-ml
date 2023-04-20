import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from normalization import max_min_normalization
from sklearn.model_selection import train_test_split
from dataClean import initData
from dataClean import *
from regression import *
from DataPreprocessing import *
from util import *
from classification import classificate

warnings.filterwarnings('ignore')

# 读文件
file = pd.read_excel("database\OUT_GCMC_fur_tip5.xlsx")

# np禁止科学计数法显示
# np.set_printoptions(suppress=True,   precision=20, threshold=10,  linewidth=40)
# pd禁止科学计数法显示
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
labels = ['type']

dataset = file[feature]

# 清洗
for item in feature:
    dataset = data_clean_delete(dataset, item)

# 74

dataset['type'] = [1]*37 +[0]*37


dataset_select = dataset[feature]
dataset_labels = dataset[labels]

X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)

classificate(X_train,Y_train,X_test,Y_test, dataset_select)

# 回归分析散点图，展示不同变量之间的关系

# for i in labels:
#     whatData(features=feature, label=i, dataset=dataset)

# # #目标值可视化
# plt.plot(dataset['Heat_furfural'])
# plt.show()

# plt.plot(dataset['Henry_furfural'])
# plt.show()

# 移除低方差特征
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(dataset_select)

# 真正想训练的内容
# for i in labels:
#     true_label = [i]
#     dataset_select = dataset[feature]
#     dataset_labels = dataset[true_label]
#     X_train, X_test, Y_train, Y_test = preprocessing(
#         dataset_select, dataset_labels, test_size=0.2)
#     singleRA(true_label, "model_RandomForestRegressor",
#              X_train, Y_train, X_test, Y_test, dataset_select)
# singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)


# 真正想训练的内容
# true_label = ['P1e+07_Con0_mg/g']
# dataset_select = dataset[feature]
# dataset_labels = dataset[true_label]
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
# singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)
