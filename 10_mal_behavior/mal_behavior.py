#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tflearn.data_utils import VocabularyProcessor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import xgboost

cmdlines_file = '../data/masquerade-data/User7'
label_file = '../data/masquerade-data/label.txt'

# np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
max_features = 128
max_document_length = 160


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

    return v, list(y)


def train_test_split(n):
    x, y = load_data()
    import random
    index = list(range(len(x)))
    random.seed(0)
    rindex = random.sample(index, len(x))
    x = [x[i] for i in rindex]
    y = [y[i] for i in rindex]

    x_train = x[:n]
    x_test = x[n:]
    y_train = y[0:n]
    y_test = y[n:]
    print(len(x_train))
    print(len(x_test))
    return x_train, x_test, y_train, y_test


def do_tfidf(n):
    # x_train, x_test, y_train, y_test = train_test_split(120)
    # print(x_train[0:2], x_test[0:2], y_train[0:2])
    #
    # vectorizer = CountVectorizer(binary=True,
    #                              min_df=1,
    #                              max_df=1,
    #                              max_features=max_features,
    #                              stop_words='english',
    #                              decode_error='ignore',
    #                              strip_accents='ascii')
    #
    # x_train = vectorizer.fit_transform(x_train)
    # x_train = x_train.toarray()
    # vocabulary = vectorizer.vocabulary_
    # print(x_train.shape)
    #
    # vectorizer = CountVectorizer(binary=True,
    #                              min_df=1,
    #                              max_df=1,
    #                              vocabulary=vocabulary,
    #                              stop_words='english',
    #                              decode_error='ignore',
    #                              strip_accents='ascii')
    #
    # x_test = vectorizer.fit_transform(x_test)
    # x_test = x_test.toarray()
    #
    # tfidf = TfidfTransformer(smooth_idf=False)
    # x_train = tfidf.fit_transform(x_train)
    # x_train = x_train.toarray()
    #
    # x_test = tfidf.transform(x_test)
    # x_test = x_test.toarray()
    # print(x_train.shape, x_train[-1, :])
    x, y = load_data()
    vectorizer = CountVectorizer(min_df=1,
                                 max_df=20,
                                 max_features=max_features,
                                 stop_words='english',
                                 decode_error='ignore',
                                 strip_accents='ascii')

    x = vectorizer.fit_transform(x)
    x = x.toarray()
    # print(x[0:8])

    tfidf = TfidfTransformer(smooth_idf=False)
    x = tfidf.fit_transform(x)
    x = x.toarray()
    print(x.shape, x[0:8])

    data = np.column_stack((x, y))
    np.random.shuffle(data)
    # import random
    # index = list(range(len(x)))
    # random.seed(0)
    # rindex = random.sample(index, len(x))
    # x = [x[i] for i in rindex]
    # y = [y[i] for i in rindex]
    x = data[:, :-1]
    y = data[:, -1]
    x_train = x[:n, :]
    x_test = x[n:, :]
    y_train = y[0:n]
    y_test = y[n:]
    print(len(x_train))
    print(len(x_test))

    return x_train, x_test, y_train, y_test


def do_n_gram(n):
    x, y = load_data()
    vectorizer = CountVectorizer(ngram_range=(2, 4),
                                 token_pattern=r'\b\w+\b',
                                 min_df=1,
                                 max_df=1.0,
                                 max_features=max_features,
                                 stop_words='english',
                                 decode_error='ignore',
                                 strip_accents='ascii')

    x = vectorizer.fit_transform(x)
    x = x.toarray()

    tfidf = TfidfTransformer(smooth_idf=False)
    x = tfidf.fit_transform(x)
    x = x.toarray()
    print(x.shape, x[0:8])

    data = np.column_stack((x, y))
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]
    x_train = x[:n, :]
    x_test = x[n:, :]
    y_train = y[0:n]
    y_test = y[n:]
    print(len(x_train))
    print(len(x_test))

    return x_train, x_test, y_train, y_test


def do_bayes(x_train, x_test, y_train, y_test):
    print('bayes')
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.precision_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


def do_mlp(x_train, x_test, y_train, y_test):
    print('mlp')
    clf = MLPClassifier(hidden_layer_sizes=(5, 2),
                        solver='lbfgs',
                        alpha=1e-5,
                        random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


def do_xgb(x_train, x_test, y_train, y_test):
    print('xgb')
    clf = xgboost.XGBClassifier(max_depth=10, learning_rate=1e-5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':
    # load_data()
    # train_test_split(53)

    print('tf-idf')
    x_train, x_test, y_train, y_test = do_tfidf(120)
    do_bayes(x_train, x_test, y_train, y_test)  # 0.7333333333333333
    do_mlp(x_train, x_test, y_train, y_test)  # 0.9
    do_xgb(x_train, x_test, y_train, y_test)  # 0.9666666666666667

    print('2,3,4-Gram&tf-idf')
    x_train, x_test, y_train, y_test = do_n_gram(120)
    do_bayes(x_train, x_test, y_train, y_test)  # 1.0
    do_mlp(x_train, x_test, y_train, y_test)  # 0.9666666666666667
    do_xgb(x_train, x_test, y_train, y_test)  # 0.9666666666666667
