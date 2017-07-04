#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/4
# Description: 决策树

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_data(n):
    """给出一个随机产生的数据集

    :param n: 数据集容量
    :return: 返回一个元组，元素依次为：训练样本集、测试样本集和各自对应的值(X_train, X_test, y_train, y_test)
    """
    np.random.seed(0)
    X = 5*np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)  # 每隔5个点添加一个随机噪声
    y[::5] += 3*(0.5-np.random.rand(noise_num))
    return train_test_split(X, y, test_size=0.25, random_state=1)


def test_DecisionTreeRegressor(*data):
    X_train, X_test, y_train, y_test = create_data()