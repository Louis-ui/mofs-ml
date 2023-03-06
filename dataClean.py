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

# 选择
feature = ['LCD','PLD','LFPD','cm3_g','ASA_m2_cm3','ASA_m2_g','AV_VF','AV_cm3_g']
labels = ['Henry_furfural','Henry_Tip5p','Heat_furfural','Heat_Tip5p']
dataset = file[feature + labels]
dataset_labels = file[labels]

# print(dataset.info())

# print(dataset_labels.info())

# 训练集测试集划分
random_state = 42
Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset, dataset_labels, test_size=0.25, random_state = random_state)



# 流水线清理数据
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(feature)),
    ('simple_imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline)
    ])

dataset_select_prepared = full_pipeline.fit_transform(Xtrain)



