# -*- coding: utf-8 -*-
import inspect
import math
import numpy as np
from sklearn import preprocessing
 
 
def max_min_normalization(data_list):
    """
    利用最大最小数将一组数据进行归一化输出
    x_new = (x - min) / (max - min)
    :param data_list:
    :return:
    """
    normalized_list = []
    max_min_interval = max(data_list) - min(data_list)
    for data in data_list:
        data = float(data)
        new_data = (data - min(data_list)) / max_min_interval
        normalized_list.append(round(new_data, 3))
 
    return normalized_list
 
 
def mean_normalization(data_list):
    """
    利用平均数将一组数据进行标准化输出
    标准化的结果不一定是在0,1之间
    x_new = (x - mean) / (max - min)
    :param data_list:
    :return:
    """
    normalized_list = []
    mean = sum(data_list) / len(data_list)
    max_min_interval = max(data_list) - min(data_list)
    for data in data_list:
        data = float(data)
        new_data = (data - mean) / max_min_interval
        normalized_list.append(round(new_data, 3))
 
    return normalized_list
 
 
def zscores_normalization(data_list):
    """
    利用z-scores方法针对数据进行标准化
    :param data_list:
    :return:
    """
    normalized_list = []
    mean = sum(data_list, 0.0) / len(data_list)
    var_lst = []
    for data in data_list:
        var_lst.append((float(data) - mean) ** 2)
    std_value = math.sqrt(sum(var_lst) / len(var_lst))
 
    for data in data_list:
        normalized_list.append(round((data - mean) / std_value, 3))
 
    return normalized_list
 
 
def max_min_normalization_using_numpy(data_list):
    """
    用数据处理包numpy归一化
    :param data_list:
    :return:
    """
    normalized_list = []
    max = np.max(data_list)
    min = np.min(data_list)
    for data in data_list:
        new_data = (float(data) - min) / (max - min)
        normalized_list.append(round(new_data, 3))
 
    return normalized_list
 
 
def zscores_normalization_using_numpy(data_list):
    """
    利用numpy中现有的方法计算标准差和平均数，然后用z-scores方法针对数据进行标准化
    :param data_list:
    :return:
    """
    normalized_list = []
    mean = np.mean(data_list)
    std = np.std(data_list)
    for data in data_list:
        normalized_list.append(round((data - mean) / std, 3))
    return normalized_list
 
 
def normalize_data_using_sk(data_list):
    """
    利用sklearn学习库自带的归一方法实现
    :param data_list:
    :return:
    """
    data_array = np.asarray(data_list, 'float').reshape(1, -1)
    new_data = preprocessing.minmax_scale(data_array, axis=1)
    return np.round(new_data, 3)[0, :]
 
 
if __name__ == '__main__':
    data_list = np.random.randint(1, 20, 10)
    data = globals().copy()
    for key in data:
        if inspect.isfunction(data[key]):
            res = data[key](data_list)
            print (key, res)