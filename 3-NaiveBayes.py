#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/6
# Description: 朴素贝叶斯

from sklearn import datasets, model_selection, naive_bayes
import numpy as np
import matplotlib.pyplot as plt


def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print ('Vector from images 0:', digits.data[0])
    for i in xrange(25):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(digits.images[i], interpolation='nearest')
    plt.show()


def load_data():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


def test_GaussianNb(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train, y_train)
    print 'Training Score:%.2f' % cls.score(X_train, y_train)
    print 'Testing Score:%.2f' % cls.score(X_test, y_test)


def test_MultinomialNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train, y_train)
    print 'Training Score:%.2f' % cls.score(X_train, y_train)
    print 'Testing Score:%.2f' % cls.score(X_test, y_test)


def test_MultinomialNB_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    # 绘图
    plt.plot(alphas, train_scores)
    plt.plot(alphas, test_scores)
    plt.legend(['Training Score', 'Testing Score'], loc='upper right')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('score')
    plt.ylim(0, 1.0)
    plt.title('MultinomialNB')
    plt.xscale('log')
    plt.show()


def test_BernoullinNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.BernoulliNB()
    cls.fit(X_train, y_train)
    print 'Training Score:%.2f' % cls.score(X_train, y_train)
    print 'Testing Score:%.2f' % cls.score(X_test, y_test)


def test_BernoulliNB_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    # 绘图
    plt.plot(alphas, train_scores)
    plt.plot(alphas, test_scores)
    plt.legend(['Training Score', 'Testing Score'], loc='best')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('score')
    plt.ylim(0, 1.0)
    plt.title('BernoullinNB')
    plt.xscale('log')
    plt.show()


def test_BernoulliNB_binaryize(*data):
    X_train, X_test, y_train, y_test = data
    min_x = min(np.min(X_train.ravel()), np.min(X_test.ravel()))-0.1
    max_x = max(np.max(X_train.ravel()), np.max(X_test.ravel()))+0.1
    binarizes = np.linspace(min_x, max_x, endpoint=True, num=100)
    train_scores = []
    test_scores = []
    for binarize in binarizes:
        cls = naive_bayes.BernoulliNB(binarize=binarize)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    # 绘图
    plt.plot(binarizes, train_scores)
    plt.plot(binarizes, test_scores)
    plt.legend(['Training Score', 'Testing Score'], loc='upper right')
    plt.xlabel('binarize')
    plt.ylabel('score')
    plt.ylim(0, 1.0)
    plt.xlim(min_x-1, max_x+1)
    plt.title('BernoulliNB')
    plt.show()


if __name__ == '__main__':
    # show_digits()
    X_train, X_test, y_train, y_test = load_data()
    # test_GaussianNb(X_train, X_test, y_train, y_test)
    # test_MultinomialNB(X_train, X_test, y_train, y_test)
    # test_MultinomialNB_alpha(X_train, X_test, y_train, y_test)
    # test_BernoullinNB(X_train, X_test, y_train, y_test)
    # test_BernoulliNB_alpha(X_train, X_test, y_train, y_test)
    test_BernoulliNB_binaryize(X_train, X_test, y_train, y_test)