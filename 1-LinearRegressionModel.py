#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/6/28
# Description:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split

def load_data():
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def test_LinerRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression(normalize=True, copy_X=False)
    regr.fit(X_train, y_train)
    print 'Cofficient:%s, intercept %.2f' % (regr.coef_, regr.intercept_)
    print 'Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2)
    print 'Score: %.2f' % regr.score(X_test, y_test)
    # regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print 'Cofficient:%s, intercept %.2f' % (regr.coef_, regr.intercept_)
    print 'Residual sum of squares: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2)
    print 'Score: %.2f' % regr.score(X_test, y_test)


def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('score')
    ax.set_xscale('log') # Set the scaling of the x-axis.
    ax.set_title('Ridge')
    plt.show()


def test_Lasso(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print 'Coefficients:%s. intercept:%.2f' % (regr.coef_, regr.intercept_)
    print 'Residual sum of squares:%.2f' % np.mean((regr.predict(X_test) - y_test) ** 2)
    print 'Score:%.2f' % regr.score(X_test, y_test)


def test_Lasso_alapha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'score')
    ax.set_xscale('log')
    ax.set_title('Lasso')
    plt.show()


def test_ElasticNet(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print 'Coefficient:%s, intercept:%.2f' % (regr.coef_, regr.intercept_)
    print 'Residual sum of squares:%.2f' % np.mean((regr.predict(X_test) - y_test) ** 2)
    print 'Score:%.2f' % regr.score(X_test, y_test)


def test_ElasticNet_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
            regr.fit(X_train, y_train)
            scores.append(regr.score(X_test, y_test))
    # 绘图
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.get_cmap(), linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\rho$')
    ax.set_zlabel('score')
    ax.set_title('ElasticNet')
    plt.show()


def load_data_iris():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train) # 分层抽样


def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    # 权重向量和b值
    print 'Cofficient:%s, intercept:%s' %(regr.coef_, regr.intercept_)
    print 'Score:%.2f' % regr.score(X_test, y_test)


def test_LogisticRegression_mutinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, y_train)
    print 'Cofficient:%s, intercept:%s' % (regr.coef_, regr.intercept_)
    print 'Score:%.2f' % regr.score(X_test, y_test)


def test_LogisticRegression_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2, 4, num=100)
    scores = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel(r'C')
    ax.set_ylabel('score')
    ax.set_xscale('log')
    ax.set_title('LogisticRegression')
    plt.show()


def test_LinearDiscriminantAnalysis(*data):
    X_train, X_test, y_train, y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print 'Cofficients:%s, intercept:%s' % (lda.coef_, lda.intercept_)
    print 'Score:%.2f' % lda.score(X_test, y_test)


def plot_LDA(converted_X, y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    makers = 'o*s'
    for target, color, maker in zip([0, 1, 2], colors, makers):
        pos = (y==target).ravel()
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=maker, label='Label %d' % target)
    ax.legend(loc='best')
    fig.suptitle('Iris After LDA')
    plt.show()


def test_LinearDiscriminantAnalysis_solver(*data):
    X_train, X_test, y_train, y_test = data
    solvers = ['svd', 'lsqr', 'eigen']
    for solver in solvers:
        if(solver=='svd'):
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver)
        else:
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver, shrinkage=None)
        lda.fit(X_train, y_train)
        print 'Score at solver=%s:%.2f' % (solver, lda.score(X_test, y_test))

def test_LinearDiscriminantAnalysis_shrinkage(*data):
    X_train, X_test, y_train, y_test = data
    shrinkages = np.linspace(0.0, 1.0, num=20)
    scores = []
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
        lda.fit(X_train, y_train)
        scores.append(lda.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shrinkages, scores)
    ax.set_xlabel(r'shrinkage')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.05)
    ax.set_title('LinearDiscriminantAnalysis')
    plt.show()

if __name__=='__main__':
    # X_train, X_test, y_train, y_test = load_data()
    # test_LinerRegression(X_train, X_test, y_train, y_test)
    # test_Ridge(X_train, X_test, y_train, y_test)
    # test_Lasso(X_train, X_test, y_train, y_test)
    # test_Lasso_alapha(X_train, X_test, y_train, y_test)
    # test_ElasticNet(X_train, X_test, y_train, y_test)
    # test_ElasticNet_alpha(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = load_data_iris()
    # test_LogisticRegression(X_train, X_test, y_train, y_test)
    # test_LogisticRegression_mutinomial(X_train, X_test, y_train, y_test)
    # test_LogisticRegression_C(X_train, X_test, y_train, y_test)
    # test_LinearDiscriminantAnalysis(X_train, X_test, y_train, y_test)

    X = np.vstack((X_train, X_test))
    Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X, Y)
    converted_X = np.dot(X, np.transpose(lda.coef_))+lda.intercept_
    plot_LDA(converted_X, Y)

    # test_LinearDiscriminantAnalysis_solver(X_train, X_test, y_train, y_test)
    # test_LinearDiscriminantAnalysis_shrinkage(X_train, X_test, y_train, y_test)