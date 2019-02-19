#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import xgboost
import gensim
import numpy as np
import multiprocessing
import os
from collections import namedtuple

max_features = 900
max_document_length = 160
vocabulary = None
doc2ver_bin="smsdoc2ver.bin"
word2ver_bin="smsword2ver.bin"
SentimentDocument = namedtuple('SentimentDocument', 'words tags')


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


def cleanText(corpus):
    '''
    处理特殊符号
    :param corpus:
    :return:
    '''
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]  # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
    return corpus


def buildWordVector(imdb_w2v,text, size):
    vec = np.zeros((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def word2vec():
    global max_features
    x_train, x_test, y_train, y_test = load_data()
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    x = x_train + x_test
    cores = multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print("find cache file %s" % word2ver_bin)
        model = gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=8, min_count=1, iter=60, workers=cores)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)

    x_train= np.concatenate([buildWordVector(model,z, max_features) for z in x_train])
    x_train = preprocessing.scale(x_train)  # 标准化， 去均值方差
    x_test= np.concatenate([buildWordVector(model,z, max_features) for z in x_test])
    x_test = preprocessing.scale(x_test)

    return x_train, x_test, y_train, y_test


def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(SentimentDocument(v, [label]))
    # print(labelized[0:2])
    #　[SentimentDocument(words=['hi', ':', ')', 'cts', 'employee', 'how', 'are', 'you', '?'], tags=['TRAIN_0']), SentimentDocument(words=['sorry', 'pa', ',', 'i', 'dont', 'knw', 'who', 'ru', 'pa', '?'], tags=['TRAIN_1'])]
    return labelized


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs), dtype='float')


def doc2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_data()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x = x_train+x_test
    cores = multiprocessing.cpu_count()

    if os.path.exists(doc2ver_bin):
        print("Find cache file %s" % doc2ver_bin)
        model=gensim.models.Doc2Vec.load(doc2ver_bin)
    else:
        model=gensim.models.Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores, iter=60)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    x_test=getVecs(model, x_test, max_features)
    x_train=getVecs(model, x_train, max_features)

    return x_train, x_test, y_train, y_test


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
    # a = 'nihao hello,op:women!lopp?==;'
    # print(a.split())  # ['nihao', 'hello,op:women!lopp?==;']
    # a = cleanText(['nihao hello,op:women!lopp?==;'])
    # print(a)

    # print('word2vec')
    # x_train, x_test, y_train, y_test = word2vec()
    # print(x_train.shape)
    # do_bayes(x_train, x_test, y_train, y_test)  # 0.9778840406455469   max_features = 900
    # do_mlp(x_train, x_test, y_train, y_test)   # 0.9790794979079498   max_features = 900
    # do_xgb(x_train, x_test, y_train, y_test)   # 0.968320382546324  max_features = 900
    #
    # do_svm(x_train, x_test, y_train, y_test)   # 0.8553496712492529 max_features = 900  效果差

    print('doc2vec')
    x_train, x_test, y_train, y_test = doc2vec()
    print(x_train.shape)
    do_bayes(x_train, x_test, y_train, y_test)  # 0.9719067543335326   max_features = 900
    do_mlp(x_train, x_test, y_train, y_test)  # 0.8553496712492529   max_features = 900  效果差
    do_xgb(x_train, x_test, y_train, y_test)  # 0.9707112970711297  max_features = 900

    do_svm(x_train, x_test, y_train, y_test)  # 0.9659294680215182 max_features = 900