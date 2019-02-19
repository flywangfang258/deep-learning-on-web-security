#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import svm


max_features = 15000


webshell_dir="../data/webshell/webshell/PHP/"
whitefile_dir="../data/webshell/normal/php/"

white_count = 0
black_count = 0


def load_one_file(file_path):
    t = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.strip('\r')
                t += line
    except:
        print(file_path)
    return t


def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        # print(d, filelist)
        for filename in filelist:
            # print(os.path.join(path, filename))
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
                print("Load %s" % fulepath)
                t = load_one_file(fulepath)
                files_list.append(t)

    return files_list


def bag_tfidf():
    global white_count
    global black_count
    global max_features
    print("max_features=%d" % max_features)
    x = []
    y = []

    webshell_files_list = load_files_re(webshell_dir)
    y1 = [1] * len(webshell_files_list)
    black_count = len(webshell_files_list)

    wp_files_list = load_files_re(whitefile_dir)
    y2 = [0] * len(wp_files_list)

    white_count = len(wp_files_list)

    x = webshell_files_list + wp_files_list
    y = y1 + y2

    CV = CountVectorizer(ngram_range=(2, 4), decode_error="ignore", max_features=max_features,
                         token_pattern=r'\b\w+\b', min_df=1, max_df=1.0)
    x = CV.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x)
    x = x_tfidf.toarray()

    return x, y


def do_metrics(y_test,y_pred):
    print("accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("confusion:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("precision:", metrics.precision_score(y_test, y_pred))
    print("recall:", metrics.recall_score(y_test, y_pred))
    print("f1:", metrics.f1_score(y_test,y_pred))


def do_mlp(x,y):
    print("mlp")
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #print clf
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)


def do_nb(x,y):
    print("nb")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)


def do_svm(x,y):
    print("svm")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    do_metrics(y_test,y_pred)


def do_rf(x,y):
    print("rf")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    do_metrics(y_test,y_pred)


def do_xgboost(x,y):

    from xgboost import XGBClassifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print("xgboost")
    xgb_model = XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    do_metrics(y_test, y_pred)


if __name__ == '__main__':
    # webshell_files_list = load_files_re(webshell_dir)
    # wp_files_list = load_files_re(whitefile_dir)
    x, y = bag_tfidf()
    # #mlp
    # do_mlp(x, y)
    # # accuracy: 0.9889380530973452
    # # confusion:
    # # [[236   2]
    # #  [3 211]]
    # # precision: 0.9906103286384976
    # # recall: 0.985981308411215
    # # f1: 0.9882903981264637
    #
    # # nb
    # do_nb(x, y)
    # # accuracy: 0.8539823008849557
    # # confusion:
    # # [[236   2]
    # #  [64 150]]
    # # precision: 0.9868421052631579
    # # recall: 0.7009345794392523
    # # f1: 0.8196721311475409
    # do_rf(x, y)
    # # accuracy: 0.9778761061946902
    # # confusion:
    # # [[228  10]
    # #  [  0 214]]
    # # precision: 0.9553571428571429
    # # recall: 1.0
    # # f1: 0.9771689497716896
    # # svm
    # do_svm(x, y)
    # # accuracy: 0.47345132743362833
    # # confusion:
    # # [[  0 238]
    # #  [  0 214]]
    # # precision: 0.47345132743362833
    # # recall: 1.0
    # # f1: 0.6426426426426427

    do_xgboost(x, y)
    # accuracy: 0.9734513274336283
    # confusion:
    # [[227  11]
    #  [1 213]]
    # precision: 0.9508928571428571
    # recall: 0.9953271028037384
    # f1: 0.9726027397260274
