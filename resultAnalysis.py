from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regressionAnalysis(method, target, y_true, y_pred):

    print('%s训练的结果' % target[0])
    print('使用%s训练' % method)
    RMSE = mean_squared_error(y_true,y_pred,squared=False)
    print('均方根误差RMSE: %f' % RMSE)
 
    MSE = mean_squared_error(y_true,y_pred)
    print('均方误差MSE: %f' % MSE)
 
    MAE = mean_absolute_error(y_true,y_pred)
    print('平均绝对误差MAE: %f' % MAE)
 
    R2 = r2_score(y_true,y_pred)
    print('R2: %f' % R2)