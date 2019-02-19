#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from hmmlearn import hmm
import numpy as np
from tflearn.data_utils import VocabularyProcessor
from sklearn import metrics
import joblib

cmdlines_file = '../data/masquerade-data/User7'
label_file = '../data/masquerade-data/label.txt'

# np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

max_document_length = 100


# 状态个数
N = 4
# 最大似然概率阈值
T = -50


def load_data():
    x = np.loadtxt(cmdlines_file, dtype=str)
    x = x.reshape((150, 100))

    y = np.loadtxt(label_file, dtype=int, usecols=6)
    y = y.reshape((100, 1))
    y_train = np.zeros((50, 1), int)
    y = np.concatenate([y_train, y])
    y = y.reshape((150, ))
    print(x.shape, y.shape)
    # print(x[0:2])
    v = []
    for k in list(x):
        v.append(" ".join(k))
    # print(v[0:5])

    return v[51:], list(y)[51:]


def load_normal_data():
    x = np.loadtxt(cmdlines_file, dtype=str)
    x = x.reshape((150, 100))

    y = np.loadtxt(label_file, dtype=int, usecols=6)
    y = y.reshape((100, 1))
    y_train = np.zeros((50, 1), int)
    y = np.concatenate([y_train, y])
    y = y.reshape((150, ))
    print(x.shape, y.shape)
    # print(x[0:2])
    v = []
    for k in list(x):
        v.append(" ".join(k))
    # print(v[0:5])

    return v[0:51], list(y)[0:51]


def do_vocabulary_table():
    x, y = load_data()

    vp = VocabularyProcessor(max_document_length=max_document_length, min_frequency=0)

    x = vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))

    return x, y


def do_hmm():
    x, y = do_vocabulary_table()
    remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100, transmat_prior=1.2)
    remodel.fit(x)
    # joblib.dump(remodel, 'hmm_sea.m')
    print('model have been saved')

    return remodel


def do_test_hmm():
    x, y = load_data()
    y_predict = []
    remodel = do_hmm()
    # print(remodel.covars_)
    print(remodel.means_)
    print(remodel.transmat_, remodel.transmat_.shape)
    print(remodel.transmat_prior)
    for sample in x[1:10]:
        y_pred = remodel.score(sample)
        print(y_pred)
    if y_pred < T:
        # 与正常行为基线的相似度小于T时，认为是疑似异常
        y_predict.append(1)
    else:
        y_predict.append(0)

    print(metrics.classification_report(y, y_predict))
    print(metrics.confusion_matrix(y, y_predict))


if __name__ == '__main__':
    do_test_hmm()
    












