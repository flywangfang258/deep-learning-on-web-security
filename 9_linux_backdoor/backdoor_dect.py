#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import glob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import xgboost

max_features = 1000


def load_attack_data():
    x = []
    y = []
    files = glob.glob('../data/ADFA-LD/Attack_Data_Master/*/*')
    for file in files:
        line = open(file).readlines()
        # print(line)
        x.append(''.join(line))
        y.append(1)
    return x, y


def load_normal_data():
    x = []
    y = []
    files = glob.glob('../data/ADFA-LD/Training_Data_Master/*')
    for file in files:
        line = open(file).readlines()
        # print(line)
        x.append(''.join(line))
        y.append(0)
    return x, y


def get_n_gram():
    x_a, y_a = load_attack_data()
    print(len(x_a))
    x_n, y_n = load_normal_data()
    print(len(x_n))
    x = x_a + x_n
    y = y_a + y_n
    print(x[0:2])

    vectorizer = CountVectorizer(ngram_range=(3, 3),
                                 token_pattern=r'\b\d+\b',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_features=max_features,
                                 max_df=1.0,
                                 min_df=1)

    x = vectorizer.fit_transform(x)

    transformer = TfidfTransformer(smooth_idf=False)
    x = transformer.fit_transform(x)
    x = x.toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

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
    # load_attack_data()

    print('3-Gram&tf-idf')
    x_train, x_test, y_train, y_test = get_n_gram()
    do_bayes(x_train, x_test, y_train, y_test)  # 0.919831223628692  max_features = 1000
    do_mlp(x_train, x_test, y_train, y_test)   # 0.9535864978902954  max_features = 1000
    do_xgb(x_train, x_test, y_train, y_test)   # 0.9071729957805907  max_features = 1000







