# 观察两表以便合并
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from DataPreprocessing import *
from dataClean import data_wash
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from regression import *
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df1 = pd.read_excel("database\data.xlsx")
df2 = pd.read_csv("database\output.csv")

# 删除后缀名.cif
df2['MOF'] = df2['MOF'].str.replace('.cif', '')

# 合并两个表
merged = pd.merge(df1, df2, left_on='filename', right_on='MOF', how='inner')
merged.to_excel('database\merged.xlsx', index=False)
Name = pd.read_excel('database\merged.xlsx')
df = Name[['filename', 'LCD', 'PLD', 'LFPD', 'cm3_g', 'ASA_m2_cm3',
           'ASA_m2_g', 'AV_VF', 'AV_cm3_g', 'Henry_furfural', 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p',
          'MOF', ' H', 'C', 'N', 'Zn', 'Cu', 'Cd', 'Co', 'Mn',
           'metal type', ' total degree of unsaturation', 'metalic percentage',
           ' oxygetn-to-metal ratio', 'electronegtive-to-total ratio', ' weighted electronegativity per atom', ' nitrogen to oxygen ']]

# 化学描述符中元素类型改成元素个数
symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
           'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

counts = df['metal type'].str.count('|'.join(symbols))

df['symbols_counts'] = counts

# 将对象类型更改为浮动类型
df = df.copy()
df['Henry_Tip5p'] = df['Henry_Tip5p'].astype(float)
df['Heat_Tip5p'] = df['Heat_Tip5p'].astype(float)
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

dataset = df

# 初始化
feature = ['LCD', 'PLD', 'LFPD', 'cm3_g',
           'ASA_m2_cm3', 'ASA_m2_g', 'AV_VF', 'AV_cm3_g', 'symbols_counts', ' H','C','N']
labels = ['Henry_furfural', 'Henry_Tip5p', 'Heat_furfural', 'Heat_Tip5p']

column = feature + labels

labels += ['selectivity_of_Henry']

column += ['selectivity_of_Henry']

dataset['selectivity_of_Henry'] = dataset['Henry_furfural'] / \
    dataset['Henry_Tip5p']

# 清洗
for item in labels:
    dataset = data_wash(dataset, item)

dataset_select = dataset[feature]
dataset_labels = dataset[labels]

dataset_labels['Henry_furfural'] = np.log(dataset_labels['Henry_furfural'] + 1)
dataset_labels['Henry_Tip5p'] = np.log(dataset_labels['Henry_Tip5p']+1)
dataset_labels['selectivity_of_Henry'] = np.log(
    dataset_labels['selectivity_of_Henry']+1)

# 移除低方差特征
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(dataset_select)

# # 数据划分
# X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)

# # 训练分析
# regressionAndAnalysis(labels, "model_RandomForestRegressor",
#                       X_train, Y_train, X_test, Y_test, dataset_select)

print(dataset.info)     #10098

# 真正想训练的内容
# for i in labels:
#     true_label = [i]
#     dataset_select = dataset[feature]
#     dataset_labels = dataset[true_label]
#     X_train, X_test, Y_train, Y_test = preprocessing(dataset_select, dataset_labels, test_size=0.2)
#     singleRA(true_label, "model_RandomForestRegressor", X_train,Y_train, X_test, Y_test, dataset_select)
#     # singleRAWithDiffModel(true_label, X_train,Y_train, X_test, Y_test, dataset_select)
