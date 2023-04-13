import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from regression import regressionAndAnalysis

file = pd.read_excel("database\data.xlsx")
name=file['filename']
df=file[['LCD','PLD','LFPD','cm3_g','ASA_m2_cm3','ASA_m2_g','AV_VF',
           'AV_cm3_g','Henry_furfural','Henry_Tip5p','Heat_furfural','Heat_Tip5p']]
feature_column=['LCD','PLD','LFPD','cm3_g','ASA_m2_cm3','ASA_m2_g','AV_VF',
           'AV_cm3_g']
target_column=['Henry_furfural','Henry_Tip5p','Heat_furfural','Heat_Tip5p','Henry_ratio']
df['Henry_ratio']=df['Henry_furfural']/df['Henry_Tip5p']

#数据粗清洗函数
def data_wash(data,column):
    #目标值为0
    def del_zero(data,column):
        querySe=data[(data[column]==0)].index.tolist()
        data_del=data.drop(querySe,axis=0)
        return data_del
    #目标值为None
    def del_nan(data,column):
        data_del=data.dropna(axis=0,how='any',subset=column)
        return data_del
    
    data=del_zero(data,column)
    data=del_nan(data,column)
    return data

#数据粗清洗
data=df  #清洗对象
column=target_column  #清洗条件

for i in column:
    data=data_wash(data,i)
    print(i,':',data.shape)
df_clean=data

#亨利常数对数化
df_log=df_clean.copy()
df_log['Henry_furfural']=np.log(df_log['Henry_furfural']+1)
df_log['Henry_Tip5p']=np.log(df_log['Henry_Tip5p']+1)
df_log['Henry_ratio']=np.log(df_log['Henry_ratio']+1)
df_log[target_column].describe()

#可视化函数
def plot_hist(data,column):
    plt.figure(figsize=(10,10))
    for i in column:
        plt.subplot(3,3,column.index(i)+1)
        data[i].plot(kind='hist',bins=50,grid=True)
        plt.title(i)
    plt.show()

#可视化数据分布
data=df_log #可视化对象list
column=target_column #可视化条件

plot_hist(data,column)

#数据划分
def split(X,y,test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
    return X_train,X_test,y_train,y_test
#插补特征缺失值
def imp(Xtrain,Xtest):
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(Xtrain)
    Xtrain_imp=imp.transform(Xtrain)
    Xtest_imp=imp.transform(Xtest)
    return Xtrain_imp,Xtest_imp
#特征缩放
def scale(Xtrain,Xtest):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    X_train_std=scaler.transform(Xtrain)
    X_test_std=scaler.transform(Xtest)
    return X_train_std,X_test_std
#目标值缩放
def normal_qt(ytrain,ytest):
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal',random_state=0)
    qt.fit(ytrain)
    y_train_normal=qt.transform(ytrain)
    y_test_normal=qt.transform(ytest)
    return y_train_normal,y_test_normal
def normal_pt(ytrain,ytest):
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson',standardize=False)
    pt.fit(ytrain)
    y_train_normal=pt.transform(ytrain)
    y_test_normal=pt.transform(ytest)
    return y_train_normal,y_test_normal
#目标值反缩放
def normal_qt_inv(ytrain,ytest):
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal',random_state=0)
    qt.fit(ytrain)
    y_train_normal=qt.inverse_transform(ytrain)
    y_test_normal=qt.inverse_transform(ytest)
    return y_train_normal,y_test_normal
def normal_pt_inv(ytrain,ytest):
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson',standardize=False)
    pt.fit(ytrain)
    y_train_normal=pt.inverse_transform(ytrain)
    y_test_normal=pt.inverse_transform(ytest)
    return y_train_normal,y_test_normal

#亨利常数
data=df_log
X=data[feature_column]
y=data[['Henry_furfural','Henry_Tip5p','Henry_ratio']]
X_train,X_test,y_train,y_test=split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#插补特征缺失值
X_train_imp,X_test_imp=imp(X_train,X_test)
#特征缩放
X_train_std,X_test_std=scale(X_train_imp,X_test_imp)

#目标值正态化
y_train_normal,y_test_normal=normal_qt(y_train,y_test)

#可视化正态数据
data=pd.DataFrame(y_train_normal,columns=y_train.columns) 
column=list(y_train.columns)

plot_hist(data,column)

#可视化正态数据
data=pd.DataFrame(y_test_normal,columns=y_test.columns) #可视化对象
column=list(y_test.columns)

plot_hist(data,column)


def model(Xtrain,Xtest,ytrain,ytest):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    model=RandomForestRegressor()
    model.fit(Xtrain,ytrain)
    ytest_pred=model.predict(Xtest)
    for i in range(ytest.shape[1]):
        print('第',i+1,'个目标值')
        RMSE=mean_squared_error(ytest[:,i],ytest_pred[:,i],squared=False)
        R2=r2_score(ytest[:,i],ytest_pred[:,i])
        print('RMSE:',RMSE)
        print('R2:',R2)
    return model,ytest_pred

def pred_true_plot(ytest,ytest_pred):
    x1=np.min(ytest)
    x2=np.max(ytest)
    y1=np.min(ytest_pred)
    y2=np.max(ytest_pred)
    plt.figure(figsize=(10,10))
    plt.scatter(ytest,ytest_pred)
    plt.plot([x1,x2],[y1,y2],color='red')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.show()


Xtrain=X_train_std
Xtest=X_test_std
ytrain=y_train_normal
ytest=y_test_normal

model,ypred=model(Xtrain,Xtest,ytrain,ytest)

#数据inverse
ytrain_inv,ytest_inv=normal_qt_inv(ytrain,ytest)
ytrain_inv,ypred_inv=normal_qt_inv(ytrain,ypred)

pred_true_plot(ytest,ypred)

pred_true_plot(ytest_inv,ypred_inv)