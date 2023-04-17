# 观察两表以便合并
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

df1 = pd.read_excel('database/OUT_GCMC_fur_tip5.xlsx ', engine='openpyxl')
df2 = pd.read_csv('database/output.csv')

df1.head(5)
df2.head(5)

# 删除后缀名.cif
df2['MOF'] = df2['MOF'].str.replace('.cif', '')

df2.head(5)


# 合并两个表
merged = pd.merge(df1, df2, left_on= 'filename', right_on='MOF', how='inner')
merged.to_excel('database/merged-test.xlsx', index=False)
Name = pd.read_excel('database/merged-test.xlsx')
df = Name[['filename', 'LCD', 'PLD', 'LFPD', 'cm3_g', 'ASA_m2_cm3',
           'ASA_m2_g', 'AV_VF', 'AV_cm3_g', 'Henry_xylose', 'Heat_xylose','Henry_Tip5p','Heat_Tip5p',
          'MOF',' H','C','N','Zn','Cu','Cd','Co','Mn',
           'metal type',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

# 从excel文件中读取数据
df.info()

# 化学描述符中元素类型改成元素个数
symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

counts = df['metal type'].str.count('|'.join(symbols))

df['symbols_counts'] = counts

# print(counts)

print(df)


#将对象类型更改为浮动类型
df = df.copy()
df['Henry_Tip5p'] = df['Henry_Tip5p'].astype(float)
df['Heat_Tip5p'] = df['Heat_Tip5p'].astype(float)
df['Henry_xylose'] = df['Henry_xylose'].astype(float)
df['Heat_xylose'] = df['Heat_xylose'].astype(float)
df[' H'] = df[' H'].astype(float)
df['C'] = df['C'].astype(float)
df['N'] = df['N'].astype(float)
df['Zn'] = df['Zn'].astype(float)
df['Cu'] = df['Cu'].astype(float)
df['Cd'] = df['Cd'].astype(float)
df['Co'] = df['Co'].astype(float)
df['Mn'] = df['Mn'].astype(float)
df['symbols_counts'] = df['symbols_counts'].astype(float)
df.dtypes


# 观察数据
df.describe()


print(df.columns)


# 计算数据框'filename'列中唯一值的数量

df['filename'].nunique()

df = df.drop(columns=['MOF'])

#数据中有多少个空值
df.isnull().sum()

print(df.shape[0])

#清洗数据
import numpy as np

#无限值（inf和-inf）替换为NaN
df = df.replace([np.inf, -np.inf], np.nan)
# #删除值为0的数据
# df = df[(df != 0).all(1)]
#删除所有重复的行
df = df.drop_duplicates()
#用中位数替换特征集中的0或者nan
features = ['LCD', 'PLD', 'LFPD', 'cm3_g', 'ASA_m2_cm3',
           'ASA_m2_g', 'AV_VF', 'AV_cm3_g',
          ' H','C','N','Zn','Cu','Cd','Co','Mn',
           'metal type',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']
df[features] = df[features].replace(0, np.nan)
df[features] = df[features].fillna(df[features].mean())
# 删除包含任何缺失值的行
df = df.dropna(axis=0, how='any')

#删除高值和低值


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
df = remove_outlier(df, 'Henry_xylose')
df = remove_outlier(df, 'Heat_xylose')
df = remove_outlier(df, 'Henry_Tip5p')
df = remove_outlier(df, 'Heat_Tip5p')
df = remove_outlier(df, ' H')
df = remove_outlier(df, 'C')
df = remove_outlier(df, 'N')
df = remove_outlier(df, 'Zn')
df = remove_outlier(df, 'Cu')
df = remove_outlier(df, 'Cd')
df = remove_outlier(df, 'Co')
df = remove_outlier(df, 'Mn')
df = remove_outlier(df, 'symbols_counts')
df = remove_outlier(df, ' total degree of unsaturation')
df = remove_outlier(df, 'metalic percentage')
df = remove_outlier(df, ' oxygetn-to-metal ratio')
df = remove_outlier(df, 'electronegtive-to-total ratio')
df = remove_outlier(df, ' weighted electronegativity per atom')
df = remove_outlier(df, ' nitrogen to oxygen ')



#删除>130000的Henry_木糖值的数据
df = df[df['Henry_xylose'] <= 130000]

#重置索引
df = df.reset_index(drop=True)

# 输入和输出
# df.to_excel('output.xlsx')

#数据中有多少个空值
df.isnull().sum()

print(df.shape[0])

#再次查看数据
df.describe()


# # 回归分析散点图，展示不同变量之间的关系
import seaborn as sns
import matplotlib.pyplot as plt

x_vars  =  ['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']

fig,  axes  =  plt.subplots(nrows=1,  ncols=len(x_vars),  figsize=(200,  5))

for  i,  ax  in  enumerate(axes):
        sns.scatterplot(x=df[x_vars[i]],    y=df['Heat_xylose'],    alpha=0.6,  ax=ax)
        ax.set_xlabel(x_vars[i],    fontsize=12)
        ax.set_ylabel('Heat_xylose',    fontsize=12)
        ax.set_title('特征关系    '    +    x_vars[i]    +    '    和    Heat_xylose',    fontsize=14)
        
plt.tight_layout()
plt.show()

x_vars  =  ['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']

fig,  axes  =  plt.subplots(nrows=1,  ncols=len(x_vars),  figsize=(200,  5))

for  i,  ax  in  enumerate(axes):
        sns.scatterplot(x=df[x_vars[i]],    y=df['Henry_xylose'],    alpha=0.6,  ax=ax)
        ax.set_xlabel(x_vars[i],    fontsize=12)
        ax.set_ylabel('Henry_xylose',    fontsize=12)
        ax.set_title('特征关系    '    +    x_vars[i]    +    '    和    Henry_xylose',    fontsize=14)
        
plt.tight_layout()
plt.show()

x_vars  =  ['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']

fig,  axes  =  plt.subplots(nrows=1,  ncols=len(x_vars),  figsize=(200,  5))

for  i,  ax  in  enumerate(axes):
        sns.scatterplot(x=df[x_vars[i]],    y=df['Heat_Tip5p'],    alpha=0.6,  ax=ax)
        ax.set_xlabel(x_vars[i],    fontsize=12)
        ax.set_ylabel('Heat_Tip5p',    fontsize=12)
        ax.set_title('特征关系    '    +    x_vars[i]    +    '    和    Heat_Tip5p',    fontsize=14)
        
plt.tight_layout()
plt.show()

x_vars  =  ['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']

fig,  axes  =  plt.subplots(nrows=1,  ncols=len(x_vars),  figsize=(200,  5))

for  i,  ax  in  enumerate(axes):
        sns.scatterplot(x=df[x_vars[i]],    y=df['Henry_Tip5p'],    alpha=0.6,  ax=ax)
        ax.set_xlabel(x_vars[i],    fontsize=12)
        ax.set_ylabel('Henry_xylose',    fontsize=12)
        ax.set_title('特征关系    '    +    x_vars[i]    +    '    和    Henry_Tip5p',    fontsize=14)
        
plt.tight_layout()
plt.show()

#目标值可视化
plt.plot(df['Heat_xylose'])
plt.show()

plt.plot(df['Heat_Tip5p'])
plt.show()

#亨利常数对数化
df['Henry_log_xylose'] = np.log10(df['Henry_xylose'] + 1)

plt.plot(df['Henry_log_xylose'])
plt.show()

df['Henry_log_water'] = np.log10(df['Henry_Tip5p'] + 1)

plt.plot(df['Henry_log_water'])
plt.show()


#将数据分成训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

def split_data(x_data, y_target, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_target, test_size=test_size, random_state=random_state)

    #正态化
    #创建QuantileTransformer对象，设置输出分布为正态分布
    quantile = QuantileTransformer(output_distribution='normal')
    #对训练集和测试集进行转换，并保存转换前的数据
    x_train_raw = x_train.copy()
    x_test_raw = x_test.copy()
    x_train = quantile.fit_transform(x_train)
    x_test = quantile.transform(x_test)
    #标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #找出离群值的索引，假设大于3或小于-3的值是离群值
    outlier_index_train = np.where(np.abs(x_train) > 3)
    outlier_index_test = np.where(np.abs(x_test) > 3)
    print("Train Outlier Index:", outlier_index_train)
    print("Test Outlier Index:", outlier_index_test)
    
    #删除原数据中对应的行
    x_train_raw = np.delete(x_train_raw, outlier_index_train[0], axis=0)
    y_train = np.delete(y_train, outlier_index_train[0], axis=0)
    x_test_raw = np.delete(x_test_raw, outlier_index_test[0], axis=0)
    y_test = np.delete(y_test, outlier_index_test[0], axis=0)
    x_train_raw  =  x_train_raw.reset_index(drop=True)
###再次转换数据
    #标准化
    scaler = StandardScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)
    
    #正态化
    x_train_raw = quantile.fit_transform(x_train_raw)
    x_test_raw = quantile.transform(x_test_raw)
    
    #归一化
    scaler = MinMaxScaler()
    x_train_raw = scaler.fit_transform(x_train_raw)
    x_test_raw = scaler.transform(x_test_raw)

    #打印数据的形状
    print('Traing_data:', x_train_raw.shape[0])
    print('Testing_data:', x_test_raw.shape[0])


    return x_train_raw, x_test_raw, y_train, y_test 


# multiple model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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
    

#   'Henry_log_xylose'
import matplotlib.pyplot as plt
x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

y_data = df['Henry_log_xylose']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test)

import matplotlib.pyplot as plt
#绘制转换后的训练集数据的直方图
plt.subplot(2, 1, 1)
plt.hist(x_train_raw, bins=30)
plt.title('训练集')

#增加子图之间的间隙
plt.subplots_adjust(hspace=0.5)
#显示图像
plt.show()
#绘制转换后的测试集数据的直方图
plt.subplot(2, 1, 1)
plt.hist(x_test_raw, bins=25)
plt.title('测试集')

#增加子图之间的间隙
plt.subplots_adjust(hspace=0.5)
#显示图像
plt.show()


#   'Heat_xylose'
x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

y_data = df['Heat_xylose']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test)


#   'Henry_log_water'
x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

y_data = df['Henry_log_water']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test)

#   'Heat_Tip5p'
x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

y_data = df['Heat_Tip5p']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

multiple_model_fit(x_train_raw, y_train, x_test_raw, y_test)



# Heat_xylose
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor

# x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
#              'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
#           'symbols_counts',' total degree of unsaturation','metalic percentage',
#            ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

# y_data = df['Heat_xylose']

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
# x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# #根据随机搜索的结果创建参数网格
# param_grid = {
#     'n_estimators': [60, 70, 80, 90, 100, 130],
#     'max_depth': [5, 9, 16, 24, 32],
#     'max_features': [0.5, 5],
#     'min_samples_split':  [2, 5, 10],  
#     'min_samples_leaf':  [1, 2, 4],  
#     'bootstrap':  [True,  False]  
# }

# # 创建基本模型
# rf = RandomForestRegressor()

# #实例化网格搜索模型
# grid_search = GridSearchCV(rf, param_grid=param_grid,
#                            cv=5, n_jobs=-1, verbose=2)

# grid_search.fit(x_train_sub, y_train_sub)

# print('BEST_PARAMS:', grid_search.best_params_)
# print('BEST_SCORE:', grid_search.best_score_)
# print('Validation accuracy: ', grid_search.score(x_val, y_val))
# print('Test accuracy: ', grid_search.score(x_test, y_test))


# # Henry_log_xylose
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor

# x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
#              'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
#           'symbols_counts',' total degree of unsaturation','metalic percentage',
#            ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

# y_data = df['Henry_log_xylose']

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
# x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# #根据随机搜索的结果创建参数网格
# param_grid = {
#     'n_estimators': [60, 70, 80, 90, 100, 130],
#     'max_depth': [5, 9, 16, 24, 32],
#     'max_features': [0.5, 5],
#     'min_samples_split':  [2, 5, 10],  
#     'min_samples_leaf':  [1, 2, 4],  
#     'bootstrap':  [True,  False]  
# }

# # 创建基本模型
# rf = RandomForestRegressor()

# #实例化网格搜索模型
# grid_search = GridSearchCV(rf, param_grid=param_grid,
#                            cv=5, n_jobs=-1, verbose=2)

# grid_search.fit(x_train_sub, y_train_sub)

# print('BEST_PARAMS:', grid_search.best_params_)
# print('BEST_SCORE:', grid_search.best_score_)
# print('Validation accuracy: ', grid_search.score(x_val, y_val))
# print('Test accuracy: ', grid_search.score(x_test, y_test))


# # Henry_log_water
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor

# x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
#              'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
#           'symbols_counts',' total degree of unsaturation','metalic percentage',
#            ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

# y_data = df['Henry_log_water']

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
# x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# #根据随机搜索的结果创建参数网格
# param_grid = {
#     'n_estimators': [60, 70, 80, 90, 100, 130],
#     'max_depth': [5, 9, 16, 24, 32],
#     'max_features': [0.5, 5],
#     'min_samples_split':  [2, 5, 10],  
#     'min_samples_leaf':  [1, 2, 4],  
#     'bootstrap':  [True,  False]  
# }

# # 创建基本模型
# rf = RandomForestRegressor()

# #实例化网格搜索模型
# grid_search = GridSearchCV(rf, param_grid=param_grid,
#                            cv=5, n_jobs=-1, verbose=2)

# grid_search.fit(x_train_sub, y_train_sub)

# print('BEST_PARAMS:', grid_search.best_params_)
# print('BEST_SCORE:', grid_search.best_score_)
# print('Validation accuracy: ', grid_search.score(x_val, y_val))
# print('Test accuracy: ', grid_search.score(x_test, y_test))


# # Heat_Tip5p
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor

# x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
#              'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
#           'symbols_counts',' total degree of unsaturation','metalic percentage',
#            ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]

# y_data = df['Heat_Tip5p']

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
# x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# #根据随机搜索的结果创建参数网格
# param_grid = {
#     'n_estimators': [60, 70, 80, 90, 100, 130],
#     'max_depth': [5, 9, 16, 24, 32],
#     'max_features': [0.5, 5],
#     'min_samples_split':  [2, 5, 10],  
#     'min_samples_leaf':  [1, 2, 4],  
#     'bootstrap':  [True,  False]  
# }

# # 创建基本模型
# rf = RandomForestRegressor()

# #实例化网格搜索模型
# grid_search = GridSearchCV(rf, param_grid=param_grid,
#                            cv=5, n_jobs=-1, verbose=2)

# grid_search.fit(x_train_sub, y_train_sub)

# print('BEST_PARAMS:', grid_search.best_params_)
# print('BEST_SCORE:', grid_search.best_score_)
# print('Validation accuracy: ', grid_search.score(x_val, y_val))
# print('Test accuracy: ', grid_search.score(x_test, y_test))


# Heat_xylose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]
y_data = df['Heat_xylose']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

# Create a based model
rf = RandomForestRegressor(
    bootstrap=True, max_depth=32, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=130, oob_score=True)

# Train the model using the training sets
rf.fit(x_train_raw, y_train)

# Predict the response for test dataset
y_pred = rf.predict(x_test_raw)

# R2 Score
print("R2_Score:", r2_score(y_test, y_pred))

# Cross Validation
scores = cross_val_score(rf, x_data, y_data, cv=5)
print('Cross-validated scores:', scores)

# Feature Importance
importances = rf.feature_importances_
features_name = x_data.columns
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(x_data.shape[1]):
   print("%d. %s (%f)" % (f + 1, features_name[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(x_data.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(x_data.shape[1]), features_name[indices], rotation=45, fontsize=8)
plt.xlim([-1, x_data.shape[1]])
plt.show()


# Henry_log_xylose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]
y_data = df['Henry_log_xylose']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

# Create a based model
rf = RandomForestRegressor(
    bootstrap=True, max_depth=24, max_features=5, min_samples_leaf=2, min_samples_split=2, n_estimators=130, oob_score=True)

# Train the model using the training sets
rf.fit(x_train_raw, y_train)

# Predict the response for test dataset
y_pred = rf.predict(x_test_raw)

# R2 Score
print("R2_Score:", r2_score(y_test, y_pred))

# Cross Validation
scores = cross_val_score(rf, x_data, y_data, cv=5)
print('Cross-validated scores:', scores)

# Feature Importance
importances = rf.feature_importances_
features_name = x_data.columns
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(x_data.shape[1]):
   print("%d. %s (%f)" % (f + 1, features_name[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(x_data.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(x_data.shape[1]), features_name[indices], rotation=45, fontsize=8)
plt.xlim([-1, x_data.shape[1]])
plt.show()


# Henry_log_water
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]
y_data = df['Henry_log_water']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

# Create a based model
rf = RandomForestRegressor(
    bootstrap=True, max_depth=24, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=100, oob_score=True)

# Train the model using the training sets
rf.fit(x_train_raw, y_train)

# Predict the response for test dataset
y_pred = rf.predict(x_test_raw)

# R2 Score
print("R2_Score:", r2_score(y_test, y_pred))

# Cross Validation
scores = cross_val_score(rf, x_data, y_data, cv=5)
print('Cross-validated scores:', scores)

# Feature Importance
importances = rf.feature_importances_
features_name = x_data.columns
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(x_data.shape[1]):
   print("%d. %s (%f)" % (f + 1, features_name[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(x_data.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(x_data.shape[1]), features_name[indices], rotation=45, fontsize=8)
plt.xlim([-1, x_data.shape[1]])
plt.show()


# Heat_Tip5p
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

x_data = df[['LCD', 'PLD', 'LFPD', 'cm3_g',
             'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g',' H','C','N','Zn','Cu','Cd','Co','Mn',
          'symbols_counts',' total degree of unsaturation','metalic percentage',
           ' oxygetn-to-metal ratio','electronegtive-to-total ratio',' weighted electronegativity per atom',' nitrogen to oxygen ']]
y_data = df['Heat_Tip5p']

x_train_raw, x_test_raw, y_train, y_test = split_data(x_data, y_data)

# Create a based model
rf = RandomForestRegressor(
    bootstrap=True, max_depth=24, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=90, oob_score=True)

# Train the model using the training sets
rf.fit(x_train_raw, y_train)

# Predict the response for test dataset
y_pred = rf.predict(x_test_raw)

# R2 Score
print("R2_Score:", r2_score(y_test, y_pred))

# Cross Validation
scores = cross_val_score(rf, x_data, y_data, cv=5)
print('Cross-validated scores:', scores)

# Feature Importance
importances = rf.feature_importances_
features_name = x_data.columns
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(x_data.shape[1]):
   print("%d. %s (%f)" % (f + 1, features_name[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(x_data.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(x_data.shape[1]), features_name[indices], rotation=45, fontsize=8)
plt.xlim([-1, x_data.shape[1]])
plt.show()