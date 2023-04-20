import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from normalization import max_min_normalization
from sklearn.model_selection import train_test_split
from dataClean import initData
from dataClean import data_wash
from regression import *
from DataPreprocessing import *
from util import *

warnings.filterwarnings('ignore')

# 读文件
file = pd.read_excel("database\OUT_GCMC_fur_tip5.xlsx")

# np禁止科学计数法显示
# np.set_printoptions(suppress=True,   precision=20, threshold=10,  linewidth=40)
# pd禁止科学计数法显示
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 去重
file.drop_duplicates(inplace=True)
file.reset_index(drop=True, inplace=True)

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
# labels = ["Hr_fur/tip5"]
labels = ['Henry_furfural', "Henry_ethanol", 'Henry_Tip5p',
          'Heat_furfural', 'Heat_Tip5p', "Hr_fur/tip5", "Hr_eth/tip5"]

column = feature + labels

dataset = file[column]

# 清洗
for item in labels:
    dataset = data_wash(dataset, item)

dataset['Henry_furfural'] = np.log(dataset['Henry_furfural']+1)
dataset['Henry_Tip5p'] = np.log(dataset['Henry_Tip5p']+1)
dataset['Henry_ethanol'] = np.log(dataset['Henry_ethanol']+1)
dataset['Hr_fur/tip5'] = np.log(dataset['Hr_fur/tip5']+1)
dataset['Hr_eth/tip5'] = np.log(dataset['Hr_eth/tip5']+1)


# 回归分析散点图，展示不同变量之间的关系

for i in labels:
    whatData(features=feature, label=i, dataset=dataset)

# #目标值可视化
plt.plot(dataset['Heat_furfural'])
plt.show()

plt.plot(dataset['Henry_furfural'])
plt.show()

# 移除低方差特征
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(dataset_select)

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


# # 数据预处理
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)

# # 训练分析
# regressionAndAnalysis(labels, "model_RandomForestRegressor",
#                       X_train, Y_train, X_test, Y_test)
