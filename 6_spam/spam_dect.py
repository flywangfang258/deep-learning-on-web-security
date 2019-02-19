#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
import tflearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.layers.embedding_ops import embedding
from tflearn.data_utils import pad_sequences, to_categorical
from tflearn import merge
import tensorflow as tf

max_features = 5000
max_document_length = 100

def demo():
    # CountVectorizer类会将文本中的词语转换为词频矩阵。
    vectorizer = CountVectorizer(min_df=1)
    # 将文本进行词袋处理
    # 语料
    corpus = [
        'This is the first document.',
        'This is the this second second document.',
        'And the third one.',
        'Is this the first document?'
    ]
    # 计算某个词出现的次数
    X = vectorizer.fit_transform(corpus)
    print(X)
    # 获取词袋中所有文本关键词，对应的特征名称
    word = vectorizer.get_feature_names()
    print(word)
    # 获取词袋数据，查看词频结果，至此已经完成了词袋化
    print(X.toarray())
    # 定义词袋的特征空间叫词汇表
    vocabulary = vectorizer.vocabulary_
    # 针对其他文本进行词袋处理时，可以直接使用现成的词汇表
    new_vectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary)


    transformer = TfidfTransformer(smooth_idf=False)
    print(transformer)

    # TF-IDF模型通常与词袋模型一起使用，对词袋模型生成的数组进行进一步的处理。
    tfidf = transformer.fit_transform(X.toarray())
    print(tfidf.toarray())


def load_one_file(filename):
    # print(filename)
    x = ''
    fr = open(filename, 'r', encoding='utf-8')
    try:
        for line in fr:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    except:
        # print(filename)
        pass
    return x


def load_files_from_dir(rootdir):
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x


def load_all_files():
    ham = []
    spam = []
    for i in range(1, 7):
        path = "../data/mail/enron%d/ham/" % i
        print("Load %s" % path)
        ham += load_files_from_dir(path)
        path = "../data/mail/enron%d/spam/" % i
        print("Load %s" % path)
        spam += load_files_from_dir(path)
    return ham, spam


def get_features_by_wordbag():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0]*len(ham)+[1]*len(spam)

    vectorizer = CountVectorizer(
                        min_df=1,
                        decode_error='ignore',
                        strip_accents='ascii',
                        stop_words='english',
                        max_df=1.0,
                        max_features=max_features
    )
    print(vectorizer)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    return x, y


def get_features_by_wordbag_tfidf():
    ham, spam = load_all_files()
    x = ham + spam
    y = [0]*len(ham) + [1]*len(spam)
    vectorizer = CountVectorizer(binary=True,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 max_features=max_features,
                                 max_df=1.0,
                                 min_df=1)
    x = vectorizer.fit_transform(x)
    x = x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x, y


def get_features_by_tf():
    global max_document_length
    ham, spam = load_all_files()
    x = ham+spam
    y = [0]*len(ham)+[1]*len(spam)
    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                min_frequency=0,
                                                vocabulary=None,
                                                tokenizer_fn=None)
    x = vp.fit_transform(x, unused_y=None)
    x = np.array(list(x))
    return x, y


def show_diffrent_max_features():
    # 最大特征数调优
    import matplotlib.pyplot as plt
    global max_features
    a = []
    b = []
    for i in range(1000, 17000, 2000):
        max_features = i
        print("max_features=%d" % i)
        x, y = get_features_by_wordbag()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        score=metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()


def do_nb(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_svm(x, y):
    x = np.array(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_mlp(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_cnn_wordbag(x_train, x_test, y_train, y_test):
    x_train = pad_sequences(x_train, maxlen=max_document_length, value=0.)
    x_test = pad_sequences(x_test, maxlen=max_document_length, value=0.)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    input_layer = input_data(shape=[None, max_document_length], name='input')
    embed = embedding(input_layer, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(embed, 128, 3, padding='valid', activation='relu', regularizer='L2')
    branch2 = conv_1d(embed, 128, 4, padding='valid', activation='relu', regularizer='L2')
    branch3 = conv_1d(embed, 128, 5, padding='valid', activation='relu', regularizer='L2')
    net = merge([branch1, branch2, branch3], mode='concat', axis=1)
    net = tf.expand_dims(net, 2)
    net = global_max_pool(net)
    dr = dropout(net, 0.8)
    fc = fully_connected(dr, 2, activation='softmax')
    net = regression(fc, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(x_train, y_train, n_epoch=3, shuffle=True, validation_set=(x_test, y_test), show_metric=True,
              batch_size=100, run_id='cnn_spam')


def do_rnn_wordbag(x_train, x_test, y_train, y_test):
    x_train = pad_sequences(x_train, maxlen=max_document_length, value=0.)
    x_test = pad_sequences(x_test, maxlen=max_document_length, value=0.)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    input = input_data([None, max_document_length], name='input')
    embed = embedding(input, input_dim=1024000, output_dim=128)
    net = lstm(embed, 128, dropout=0.8)
    fc = fully_connected(net, 2, activation='softmax')
    net = regression(fc, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, batch_size=10, run_id='lstm_spam', n_epoch=3)


if __name__ == '__main__':
    # ham, spam = load_all_files()
    # print(ham[0])
    #
    # x, y = get_features_by_wordbag()
    # print(np.array(x).shape, np.array(y).shape)
    # do_nb(x, y)   # 94%
    # do_svm(x, y)  # 90%
    # do_mlp(x, y)  # 98%
    # show_diffrent_max_features()

    x, y = get_features_by_wordbag_tfidf()
    do_nb(x, y)  # 94%
    do_svm(x, y)
    do_mlp(x, y)  # 0.9863568956994563

    print("get_features_by_tf")
    x, y = get_features_by_tf()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    # CNN
    do_cnn_wordbag(x_train, x_test, y_train, y_test)

    # RNN
    do_rnn_wordbag(x_train, x_test, y_train, y_test)











