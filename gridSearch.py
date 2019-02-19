#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score


def simpleGridSearch(X_train, X_test, y_train, y_test):
    '''
    使用for循环实现网格搜索
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    # grid search start
    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma,C=C)#对于每种参数可能的组合，进行一次训练；
            svm.fit(X_train,y_train)
            score = svm.score(X_test,y_test)
            if score > best_score:#找到表现最好的参数
                best_score = score
                best_parameters = {'gamma':gamma,'C':C}

    print("Best score:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))


def gridSearchCv(X_train, X_test, y_train, y_test):
    '''
    使用for循环实现网格搜索与交叉验证
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    best_score = 0.0
    for gamma in [0.001,0.01,0.1,1,10,100]:
        for C in [0.001,0.01,0.1,1,10,100]:
            svm = SVC(gamma=gamma, C=C)
            scores = cross_val_score(svm, X_train, y_train, cv=5) #5折交叉验证
            score = scores.mean() #取平均数
            if score > best_score:
                best_score = score
                best_parameters = {"gamma": gamma, "C": C}
    svm = SVC(**best_parameters)
    svm.fit(X_train, y_train)
    test_score = svm.score(X_test,y_test)
    print("Best score on validation set:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))
    print("Score on testing set:{:.2f}".format(test_score))


def skGridSearchCv(X_train, X_test, y_train, y_test):
    '''
    利用sklearn中的GridSearchCV类
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    #把要调整的参数以及其候选值 列出来；
    param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
                 "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    print("Parameters:{}".format(param_grid))

    grid_search = GridSearchCV(SVC(), param_grid, cv=5) # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
    print("Size of training set:{} size of testing set:{}".format(X_train.shape[0], X_test.shape[0]))
    # simpleGridSearch(X_train, X_test, y_train, y_test)
    # gridSearchCv(X_train, X_test, y_train, y_test)
    skGridSearchCv(X_train, X_test, y_train, y_test)



