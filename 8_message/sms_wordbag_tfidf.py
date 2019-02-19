#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tflearn.data_utils import VocabularyProcessor

import xgboost


max_features = 900
max_document_length = 160
vocabulary = None


def load_data():
    x = []
    y = []
    fr = open('../data/smsspamcollection/SMSSpamCollection.txt', 'r', encoding='utf-8')
    for line in fr:
        line = line.strip().split('\t', maxsplit=1)
        x.append(line[1])
        if line[0] == 'ham':
            y.append(0)
        else:
            y.append(1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    return x_train, x_test, y_train, y_test


def bag_of_words():
    global max_features
    x_train, x_test, y_train, y_test = load_data()

    vectorizer = CountVectorizer(decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=max_features)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 vocabulary=vocabulary)

    x_test = vectorizer.transform(x_test)
    x_test = x_test.toarray()
    return x_train, x_test, y_train, y_test


def tfidf_feature():
    global max_features
    x_train, x_test, y_train, y_test = load_data()

    vectorizer = CountVectorizer(binary=True,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=max_features)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(binary=True,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 vocabulary=vocabulary)

    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()

    tfidf = TfidfTransformer(smooth_idf=False)
    x_train = tfidf.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = tfidf.transform(x_test)
    x_test = x_test.toarray()

    return x_train, x_test, y_train, y_test


def vocabulary_table():
    global max_document_length
    x_train, x_test, y_train, y_test = load_data()
    vp = VocabularyProcessor(max_document_length=max_document_length,
                             min_frequency=10,
                             vocabulary=None,
                             tokenizer_fn=None)
    x_train = vp.fit_transform(x_train, unused_y=None)
    x_train = np.array(list(x_train))
    x_test = vp.transform(x_test)
    x_test = np.array(list(x_test))
    return  x_train, x_test, y_train, y_test


def do_svm(x_train, x_test, y_train, y_test):
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1

    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1

    print('svm')
    clf = SVC(C=10, gamma=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.precision_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


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
    # print('bag of words')
    # x_train, x_test, y_train, y_test = bag_of_words()
    # do_bayes(x_train, x_test, y_train, y_test)  # 0.8158995815899581   max_features = 1500
    # do_mlp(x_train, x_test, y_train, y_test)   # 0.976688583383144   max_features = 1500
    # do_xgb(x_train, x_test, y_train, y_test)   # 0.9420203227734608  max_features = 1500

    # do_svm(x_train, x_test, y_train, y_test)   # 0.9019725044829647 max_features = 900

    print('tfidf')
    x_train, x_test, y_train, y_test = tfidf_feature()
    do_bayes(x_train, x_test, y_train, y_test)  # 0.812910938433951  max_features = 1500
    do_mlp(x_train, x_test, y_train, y_test)   # 0.9689181111775254  max_features = 1500
    do_xgb(x_train, x_test, y_train, y_test)   # 0.9432157800358637  max_features = 1500

    do_svm(x_train, x_test, y_train, y_test)    # 0.982068141063957  max_features = 900

    # print('vocabulary')
    # x_train, x_test, y_train, y_test = vocabulary_table()
    # do_bayes(x_train, x_test, y_train, y_test)  # terrible
    # do_mlp(x_train, x_test, y_train, y_test)    # terrible
    # do_svm(x_train, x_test, y_train, y_test)  # 0.8840406455469217  max_document_length = 500
    # do_xgb(x_train, x_test, y_train, y_test)  # 0.885833831440526  max_document_length = 500



