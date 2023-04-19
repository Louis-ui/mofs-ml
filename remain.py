from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataClean import data_wash
from regression import regressionAndAnalysis
from DataPreprocessing import *
from resultAnalysis import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')

file = pd.read_excel("database\data.xlsx")

feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
labels = ['Henry_furfural', 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p']

column = feature + labels

dataset = file[column]

labels += ['selectivity_of_Henry']

column += ['selectivity_of_Henry']

dataset['selectivity_of_Henry'] = dataset['Henry_furfural'] / \
    dataset['Henry_Tip5p']

dataset['Henry_furfural'] = np.log(dataset['Henry_furfural'] + 1)
dataset['Henry_Tip5p'] = np.log(dataset['Henry_Tip5p']+1)
dataset['selectivity_of_Henry'] = np.log(dataset['selectivity_of_Henry']+1)

# 清洗数据

# 无限值（inf和-inf）替换为NaN
df = dataset.replace([np.inf, -np.inf], np.nan)
# #删除值为0的数据
# df = df[(df != 0).all(1)]
# 删除所有重复的行
df = df.drop_duplicates()
# 用中位数替换特征集中的0或者nan
features = ['LCD', 'PLD', 'LFPD', 'cm3_g', 'ASA_m2_cm3',
            'ASA_m2_g', 'AV_VF', 'AV_cm3_g']
df[features] = df[features].replace(0, np.nan)
df[features] = df[features].fillna(df[features].mean())
# 删除包含任何缺失值的行
df = df.dropna(axis=0, how='any')

# 删除高值和低值
def remove_outlier(data, column):
    Q1 = data[column].quantile(0.035)
    Q3 = data[column].quantile(0.965)
    IQR = Q3 - Q1
    data = data[(data[column] >= Q1 - 1.5 * IQR) &
                (data[column] <= Q3 + 1.5 * IQR)]
    return data

df = remove_outlier(df, 'LCD')
df = remove_outlier(df, 'PLD')
df = remove_outlier(df, 'LFPD')
df = remove_outlier(df, 'cm3_g')
df = remove_outlier(df, 'ASA_m2_cm3')
df = remove_outlier(df, 'ASA_m2_g')
df = remove_outlier(df, 'AV_VF')
df = remove_outlier(df, 'AV_cm3_g')
df = remove_outlier(df, 'Henry_furfural')
df = remove_outlier(df, 'Heat_furfural')
df = remove_outlier(df, 'Henry_Tip5p')
df = remove_outlier(df, 'Heat_Tip5p')

# 重置索引
df = df.reset_index(drop=True)


def split_data(x_data, y_target, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=test_size, random_state=random_state)

    # 正态化
    # 创建QuantileTransformer对象，设置输出分布为正态分布
    quantile = QuantileTransformer(output_distribution='normal')
    # 对训练集和测试集进行转换，并保存转换前的数据
    x_train_raw = x_train.copy()
    x_test_raw = x_test.copy()
    x_train = quantile.fit_transform(x_train)
    x_test = quantile.transform(x_test)
    # 标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 找出离群值的索引，假设大于3或小于-3的值是离群值
    outlier_index_train = np.where(np.abs(x_train) > 3)
    outlier_index_test = np.where(np.abs(x_test) > 3)
    print("Train Outlier Index:", outlier_index_train)
    print("Test Outlier Index:", outlier_index_test)

    #   删除原数据中对应的行
    x_train_raw = np.delete(x_train_raw, outlier_index_train[0], axis=0)
    y_train = np.delete(y_train, outlier_index_train[0], axis=0)
    x_test_raw = np.delete(x_test_raw, outlier_index_test[0], axis=0)
    y_test = np.delete(y_test, outlier_index_test[0], axis=0)
    x_train_raw = x_train_raw.reset_index(drop=True)
    # 再次转换数据
    # 标准化
    scaler = StandardScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)

    # 正态化
    x_train_raw = quantile.fit_transform(x_train_raw)
    x_test_raw = quantile.transform(x_test_raw)

    # 归一化
    scaler = MinMaxScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)

    # 打印数据的形状
    print('Traing_data:', x_train_raw.shape[0])
    print('Testing_data:', x_test_raw.shape[0])

    return x_train_raw, x_test_raw, y_train, y_test


# multiple model


def model_fit(model, x_train_raw, y_train, x_test_raw, y_test):

    model.fit(x_train_raw, y_train)
    y_pred = model.predict(x_test_raw)

    MODEL = model.__class__.__name__
    R2_SCORE = r2_score(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    print('{},R2_SCORE:{},RMSE:{}'.format(MODEL, R2_SCORE, RMSE))

    return MODEL, R2_SCORE, RMSE


def multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test):

    models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), DecisionTreeRegressor(
    ), RandomForestRegressor(), GradientBoostingRegressor(), SVR(), KNeighborsRegressor()]

    BEST_MODEL = "model"
    BEST_R2_SCORE = 0
    BEST_RMSE = 0

    for model in models:
        MODEL, R2_SCORE, RMSE = model_fit(
            model, x_train_raw, y_train, x_test_raw, y_test)
        if R2_SCORE > BEST_R2_SCORE:
            BEST_R2_SCORE = R2_SCORE
            BEST_RMSE = RMSE
            BEST_MODEL = MODEL

    print('BEST_MODEL:{}, BEST_R2_SCORE:{}, BEST_RMSE:{}'.format(
        BEST_MODEL, BEST_R2_SCORE, BEST_RMSE))


# #   'Henry_furfural'
# x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
#              'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g']]

# y_data = df['Henry_furfural']

# x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

# multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test)

# # 绘制转换后的训练集数据的直方图
# plt.subplot(2, 1, 1)
# plt.hist(x_train_raw, bins=30)
# plt.title('训练集')

# # 增加子图之间的间隙
# plt.subplots_adjust(hspace=0.5)
# # 显示图像
# plt.show()
# # 绘制转换后的测试集数据的直方图
# plt.subplot(2, 1, 1)
# plt.hist(x_test_raw, bins=25)
# plt.title('测试集')

# # 增加子图之间的间隙
# plt.subplots_adjust(hspace=0.5)
# # 显示图像
# plt.show()
