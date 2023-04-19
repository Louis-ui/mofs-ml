from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from matplotlib import rcParams
from pylab import mpl
from resultAnalysis import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
rcParams['axes.unicode_minus'] = False

models = {
    "model_LinearRegression": linear_model.LinearRegression(),
    "model_DecisionTreeRegressor": tree.DecisionTreeRegressor(),
    # "model_SVR": svm.SVR(),
    "model_KNeighborsRegressor": neighbors.KNeighborsRegressor(),
    "model_RandomForestRegressor": ensemble.RandomForestRegressor(n_estimators=20),
    # "model_AdaBoostRegressor": ensemble.AdaBoostRegressor(n_estimators=50),
    # "model_GradientBoostingRegressor": ensemble.GradientBoostingRegressor(n_estimators=100),
    "model_BaggingRegressor": BaggingRegressor(),
    "model_ExtraTreeRegressor":  ExtraTreeRegressor()
}


def regressionAndAnalysis(target_labels, method, Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare, X_data):
    model = models[method]
    model.fit(Xtrain_prepare, Ytrain_prepare)
    predictions = model.predict(Xtest_prepare)
    regressionAnalysis(method, target_labels,Ytest_prepare, predictions)
    # 特征重要程度分析
    important(X_data, model)

# def regressionAnalysis(method, target, y_true, y_pred):

#     print('%s训练的结果' % target[0])
#     print('使用%s训练' % method)
#     RMSE = mean_squared_error(y_true,y_pred,squared=False)
#     print('均方根误差RMSE: %.5f' % RMSE)
 
#     MSE = mean_squared_error(y_true,y_pred)
#     print('均方误差MSE: %.5f' % MSE)
 
#     MAE = mean_absolute_error(y_true,y_pred)
#     print('平均绝对误差MAE: %.5f' % MAE)
 
#     R2 = r2_score(y_true,y_pred)
#     print('R2: %.5f' % R2)


# def regressionAndAnalysis(target_labels, method, Xtrain_prepare, Ytrain_prepare, Xtest_prepare, Ytest_prepare):
#     for function in models:
#         model = models[function]
#         print(function)
#     # model = models[method]
#         model.fit(Xtrain_prepare, Ytrain_prepare)
#         predictions = model.predict(Xtest_prepare)
#         for index in range(Ytest_prepare.shape[1]):
#             target_label = [target_labels[index]]
#             regressionAnalysis(function, target_label, Ytest_prepare[:,index], predictions[:,index])

# model = models["model_RandomForestRegressor"]
# model.fit(Xtrain_prepare, Ytrain_prepare.ravel())
# predictions = model.predict(Xtest_prepare)

# Ytest_prepare_list = []

# for item in Ytest_prepare:
#     Ytest_prepare_list.append(item)

# regressionAnalysis(target_label[0], Ytest_prepare_list, predictions)
