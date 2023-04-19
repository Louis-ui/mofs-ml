from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def regressionAnalysis(method, target, y_true, y_pred):

    print('%s训练的结果' % target[0])
    print('使用%s训练' % method)
    RMSE = mean_squared_error(y_true,y_pred,squared=False)
    print('均方根误差RMSE: %f' % RMSE)
 
    # MSE = mean_squared_error(y_true,y_pred)
    # print('均方误差MSE: %f' % MSE)
 
    # MAE = mean_absolute_error(y_true,y_pred)
    # print('平均绝对误差MAE: %f' % MAE)
 
    R2 = r2_score(y_true,y_pred)
    print('R2: %f' % R2)


def important(x_data, model):
    # Feature Importance
    importances = model.feature_importances_
    features_name = x_data.columns
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
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