from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from dataClean import Xtrain, Ytrain, Xtest, Ytest

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
rcParams['axes.unicode_minus'] = False


models = {
    "model_LinearRegression": linear_model.LinearRegression(),
    "model_DecisionTreeRegressor": tree.DecisionTreeRegressor(),
    "model_SVR": svm.SVR(),
    "model_KNeighborsRegressor": neighbors.KNeighborsRegressor(),
    "model_RandomForestRegressor": ensemble.RandomForestRegressor(n_estimators=20),
    "model_AdaBoostRegressor": ensemble.AdaBoostRegressor(n_estimators=50),
    "model_GradientBoostingRegressor": ensemble.GradientBoostingRegressor(n_estimators=100),
    "model_BaggingRegressor": BaggingRegressor(),
    "model_ExtraTreeRegressor":  ExtraTreeRegressor()
}

print(Xtrain)
print(Ytrain)

model = models["model_LinearRegression"]
model.fit(Xtrain, Ytrain)
predictions = model.predict(Xtest)

plt.scatter(Xtrain, Ytrain, color="g")
plt.plot(Xtest, predictions, color="r")
plt.show()
