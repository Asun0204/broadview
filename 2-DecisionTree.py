#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/4
# Description: 决策树

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
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
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)
    print 'Training score:%f' % regr.score(X_train, y_train)
    print 'Testing score:%f' % regr.score(X_test, y_test)
    ## 绘图
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)
    plt.scatter(X_train, y_train, label='train sample', c='g')
    plt.scatter(X_test, y_test, label='test sample', c='r')
    plt.plot(X, Y, label='predict_value', linewidth=2, alpha=0.5)
    plt.xlabel('data')
    plt.ylabel('score')
    plt.title('Decision Tree Regression')
    plt.legend(framealpha=0.5)
    # ax.scatter(X_train, y_train, label='train sample', c='g')
    # ax.scatter(X_test, y_test, label='test sample', c='r')
    # ax.plot(X, Y, label='predict_value', linewidth=2, alpha=0.5)
    # ax.set_xlabel('data')
    # ax.set_ylabel('target')
    # ax.set_title('Decision Tree Regression')
    # ax.legend(framealpha=0.5)
    plt.show()


def test_DecisionTreeRegressor_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train, y_train)
        print 'Splitter %s' % splitter
        print 'Training score:%f' % regr.score(X_train, y_train)
        print 'Testing score:%f' % regr.score(X_test, y_test)


def test_DecisionTreeRegressor_depth(maxdepth, *data):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label='training score')
    ax.plot(depths, testing_scores, label='testing score')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    ax.set_title('Decision Tree Regression')
    ax.legend(framealpha=0.5)
    plt.show()


def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def test_DecisionTreeClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    export_graphviz(clf, 'out')

    print 'Training socre:%f' % clf.score(X_train, y_train)
    print 'Testing score:%f' % clf.score(X_test, y_test)


def test_DecisionTreeClassifier_criterion(*data):
    X_train, X_test, y_train, y_test = data
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print 'criterion:%s' % criterion
        print 'Training score:%f' % clf.score(X_train, y_train)
        print 'Testing score:%f' % clf.score(X_test, y_test)


def test_DecisionTreeClassifier_splitter(*data):
    X_train, X_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train, y_train)
        print 'splitter:%s' % splitter
        print 'Training score:%f' % clf.score(X_train, y_train)
        print 'Testing score:%f' % clf.score(X_test, y_test)


def test_DecisionTreeClassifier_depth(maxdepth, *data):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    plt.plot(depths, training_scores, label='training score', marker='o')
    plt.plot(depths, testing_scores, label='testing score', marker='*')
    plt.xlabel('maxdepth')
    plt.ylabel('score')
    plt.title('Decision Tree Classification')
    plt.legend(framealpha=0.5, loc='best')
    plt.show()


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = create_data(100)
    # test_DecisionTreeRegressor(X_train, X_test, y_train, y_test)
    # test_DecisionTreeRegressor_splitter(X_train, X_test, y_train, y_test)
    # test_DecisionTreeRegressor_depth(20, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = load_data()
    test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    # test_DecisionTreeClassifier_criterion(X_train, X_test, y_train, y_test)
    # test_DecisionTreeClassifier_splitter(X_train, X_test, y_train, y_test)
    # test_DecisionTreeRegressor_depth(100, X_train, X_test, y_train, y_test)