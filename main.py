import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataClean import data_wash
from regression import *
from DataPreprocessing import *
from resultAnalysis import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from util import whatData

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
labels = ['Henry_furfural', 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p']

column = feature + labels

dataset = file[column]

labels += ['selectivity_of_Henry']

column += ['selectivity_of_Henry']

dataset['selectivity_of_Henry'] = dataset['Henry_furfural'] / \
    dataset['Henry_Tip5p']

# 清洗
for item in labels:
    dataset = data_wash(dataset, item)

dataset['Henry_furfural'] = np.log(dataset['Henry_furfural'] + 1)
dataset['Henry_Tip5p'] = np.log(dataset['Henry_Tip5p']+1)
dataset['selectivity_of_Henry'] = np.log(dataset['selectivity_of_Henry']+1)

# # 真正想训练的内容
# for i in labels:
#     true_label = [i]
#     dataset_select = dataset[feature]
#     dataset_labels = dataset[true_label]
#     X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
#     singleRA(true_label, "model_RandomForestRegressor", X_train,Y_train, X_test, Y_test, dataset_select)
#     # singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)

# 真正想训练的内容
true_label = ['Henry_furfural']
dataset_select = dataset[feature]
dataset_labels = dataset[true_label]
X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
singleRA(true_label, "model_RandomForestRegressor", X_train,Y_train, X_test, Y_test, dataset_select)

# labels = ['selectivity_of_Henry']

# dataset_select = dataset[feature]
# dataset_labels = dataset[labels]

# 回归分析散点图，展示不同变量之间的关系

# whatData(features=feature, label='Heat_furfural', dataset=dataset)

# whatData(features=feature, label='Henry_furfural', dataset=dataset)

# #目标值可视化
# plt.plot(dataset['Heat_furfural'])
# plt.show()

# plt.plot(dataset['Henry_furfural'])
# plt.show()

# 移除低方差特征
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(dataset_select)

# # 数据预处理
# X_train, X_test, Y_train, Y_test = preprocessing(
#     dataset_select, dataset_labels, test_size=0.2)

# # 训练分析
# singleRA(labels, "model_RandomForestRegressor", X_train,
#          Y_train, X_test, Y_test, dataset_select)
