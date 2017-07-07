#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/7
# Description: k-近邻算法

import  numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, model_selection


def load_classification_data():
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def create_regression_data(n):
    X = 5*np.random.rand(n, 1)
    y = np.sin(X.ravel())
    y[::5] += 1*(0.5-np.random.rand(int(n/5)))
    return model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


def test_KNeighborsClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print 'Training Score:%.2f' % clf.score(X_train, y_train)
    print 'Testing Score:%.2f' % clf.score(X_test, y_test)


def test_KNeighborClassifier_k_w(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        testing_scores = []
        training_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=K)
            clf.fit(X_train, y_train)
            testing_scores.append(clf.score(X_test, y_test))
            training_scores.append((clf.score(X_train, y_train)))
        ax.plot(Ks, training_scores, label='testing score:weight=%s' % weight)
        ax.plot(Ks, testing_scores, label='training score:weight=%s' % weight)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsClassifier')
    plt.show()


def test_KNeighborClassifier_k_p(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for P in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(p=P, n_neighbors=K)
            clf.fit(X_train, y_train)
            testing_scores.append(clf.score(X_test, y_test))
            training_scores.append(clf.score(X_train, y_train))
        ax.plot(Ks, testing_scores, label='testing score:p=%s' % P)
        ax.plot(Ks, training_scores, label='training score:p=%s' % P)
    ax.legend(loc='best')
    ax.set_xlabel('K')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('KNeighborsClassifier')
    plt.show()


def test_KNeighborsRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = neighbors.KNeighborsRegressor()
    regr.fit(X_train, y_train)
    print 'Training Score:%f' % regr.score(X_train, y_train)
    print 'Testing Score:%f' % regr.score(X_test, y_test)


def test_KNeighborsRegressor_k_p(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    for P in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            regr = neighbors.KNeighborsRegressor(p=P, n_neighbors=K)
            regr.fit(X_train, y_train)
            testing_scores.append(regr.score(X_test, y_test))
            training_scores.append(regr.score(X_train, y_train))
        plt.plot(Ks, testing_scores)
        plt.plot(Ks, training_scores)
        plt.legend(['testing score:p=%d' % P, 'training score:p=%d' % P])
    plt.xlabel('K')
    plt.ylabel('score')
    plt.ylim(0, 1.05)
    plt.title('KNeighborsRegressor')
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_classification_data()
    # test_KNeighborsClassifier(X_train, X_test, y_train, y_test)
    # test_KNeighborClassifier_k_w(X_train, X_test, y_train, y_test)
    # test_KNeighborClassifier_k_p(X_train, X_test, y_train, y_test)
    # test_KNeighborsRegressor(X_train, X_test, y_train, y_test)
    test_KNeighborClassifier_k_p(X_train, X_test, y_train, y_test)