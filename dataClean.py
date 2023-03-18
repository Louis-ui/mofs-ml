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
from util import ExeLabelEncoder

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
# MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL = 5.7
# dataset_drop_small_LCD = dataset[dataset['LCD']>=MOLECULAR_DYNAMICS_DIAMETER_OF_FURFURAL]

# dataset_labels = dataset_drop_small_LCD[labels]
# dataset_select = dataset_drop_small_LCD[feature]

dataset_labels = dataset[labels]
dataset_select = dataset[feature]

dataset_labels['Henry_furfural'] = np.log(dataset_labels['Henry_furfural'])
dataset_labels['Henry_Tip5p'] = np.log(dataset_labels['Henry_Tip5p'])

dataset_labels['selectivity_of_Henry'] = dataset_labels.apply(lambda x: x["Henry_furfural"] / x["Henry_Tip5p"], axis=1)
dataset_labels['selectivity_of_Heat'] = dataset_labels.apply(lambda x: x["Heat_furfural"] / x["Heat_Tip5p"], axis=1)
labels.append('selectivity_of_Henry')
labels.append('selectivity_of_Heat')

# print(dataset_select.info())
# print(dataset_labels.info())

#真正需要回归的目标
target_label = ['selectivity_of_Henry']
dataset_target_labels = dataset_labels[target_label]

# 训练集测试集划分
random_state = 42
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    dataset_select, dataset_target_labels, test_size=0.25, random_state=random_state)

# print(Xtest)


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

