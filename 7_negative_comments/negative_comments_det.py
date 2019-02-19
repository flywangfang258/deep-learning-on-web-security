#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn import merge
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.embedding_ops import embedding
from tflearn.data_utils import pad_sequences, to_categorical
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tflearn.data_utils import VocabularyProcessor
import gensim
from gensim.models import Doc2Vec
import multiprocessing
from collections import namedtuple


max_features = 128
max_document_length = 1000
doc2ver_bin="doc2ver.bin"
SentimentDocument = namedtuple('SentimentDocument', 'words tags')


def word2vec_demo():
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    model = gensim.models.Word2Vec(sentences, min_count=2, size=15)
    print(model['sentence'])

    model = Doc2Vec(dm=0, dbow_words=1, size=15, window=8, min_count=1, iter=10, workers=4)
    label1 = ('Train_1')
    label2 = ('Train_2')
    x1 = SentimentDocument(sentences[0], [label1])
    x2 = SentimentDocument(sentences[1], [label2])
    x = [x1, x2]
    model.build_vocab(x)
    # model.train(x, total_examples=model.corpus_count, epochs=model.iter)
    print(model.docvecs['Train_1'])
    print(model.docvecs['Train_2'])
    print(model['first'])


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
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    path="../data/review/aclImdb/train/pos/"
    print("Load %s" % path)
    x_train += load_files_from_dir(path)
    y_train = [0]*len(x_train)
    path = "../data/review/aclImdb/train/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    x_train += tmp
    y_train += [1]*len(tmp)


    path = "../data/review/aclImdb/test/pos/"
    print("Load %s" % path)
    x_test += load_files_from_dir(path)
    y_test = [0]*len(x_test)
    path="../data/review/aclImdb/test/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    x_test += tmp
    y_test += [1]*len(tmp)
    return x_train, x_test, y_train, y_test


def get_features_by_wordbag():
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()
    # print(x_train[0:2])
    # print(type(x_train[1]))
    # print(x_test[0:2])

    vectorizer = CountVectorizer(min_df=1,
                        lowercase=False,
                        decode_error='ignore',
                        strip_accents='ascii',
                        stop_words='english',
                        max_df=1.0,
                        max_features=max_features
    )
    print(vectorizer)

    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(decode_error='ignore',
                                 strip_accents='ascii',
                                 stop_words='english',
                                 vocabulary=vocabulary,
                                 max_df=1,
                                 min_df=1)
    print(vectorizer)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()
    return x_train, x_test, y_train, y_test


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print("SVM and wordbag")
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def get_features_by_wordbag_tfidf():
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()

    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1,
        binary=True)
    print(vectorizer)

    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        vocabulary=vocabulary,
        stop_words='english',
        max_df=1.0, binary=True,
        min_df=1)
    print(vectorizer)

    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train=transformer.fit_transform(x_train)
    x_train=x_train.toarray()
    x_test=transformer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test


def get_features_by_tf():
    global max_document_length
    x_train, x_test, y_train, y_test=load_all_files()

    vp=VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x_train=vp.fit_transform(x_train, unused_y=None)
    x_train=np.array(list(x_train))

    x_test=vp.transform(x_test)
    x_test=np.array(list(x_test))
    return x_train, x_test, y_train, y_test


def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    print("MLP and wordbag")

    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_cnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print("CNN and tf")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None, max_document_length], name='input')
    network = embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")


def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    print("rf and doc2vec")
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


def get_feature_by_word2vec(word2ver_bin):
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    cores = multiprocessing.cpu_count()
    print(cores)  # 当前计算机cpu的个数

    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model = gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=8, min_count=10, iter=10, workers=cores)   # 初始化Word2Vec对象
        x = x_train + x_test
        # 创建字典并开始训练获取word2vec
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        # 经过训练后，Word2vec会以字典的形式保存在model对象中，获取 model['love']
        # word2vec的维度和之前设置的神经网络隐藏层的单元数相同，为200，即一个长度为200的一维向量。
        model.save(word2ver_bin)

    x_train = np.concatenate([buildWordVector(model, z, max_features) for z in x_train])
    x_test = np.concatenate([buildWordVector(model, z, max_features) for z in x_test])

    return x_train, x_test, y_train, y_test


def buildWordVector(imdb_w2v,text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(SentimentDocument(v, [label]))
    return labelized


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    # print(corpus[0].tags)   # ['TEST_0']  # ['TRAIN_0']
    return np.array(np.concatenate(vecs), dtype='float')


def get_features_by_doc2vec():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()
    # print(x_test[0])

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(doc2ver_bin):
        print("Find cache file %s" % doc2ver_bin)
        model=Doc2Vec.load(doc2ver_bin)
    else:
        # dbow_words:1训练词向量，0只训练doc向量。
        model = Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores, iter=60)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    # print(model.docvecs['TEST_0'])
    x_test=getVecs(model, x_test, max_features)
    x_train=getVecs(model, x_train, max_features)
    return x_train, x_test, y_train, y_test


def do_cnn_doc2vec(trainX, testX, trainY, testY):
    global max_features
    print("CNN and doc2vec")

    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    trainX = trainX.reshape([-1, max_features, 1])
    testX = testX.reshape([-1, max_features, 1])
    # Building convolutional network
    network = input_data(shape=[None, max_features, 1], name='input')
    # network = tflearn.embedding(network, input_dim=1000000, output_dim=128,validate_indices=False)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100, run_id="review")


def do_rnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10, run_id="review", n_epoch=5)


def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    print("NB and doc2vec")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    # x_train, x_test, y_train, y_test = get_features_by_wordbag()
    # # NB
    # do_nb_wordbag(x_train, x_test, y_train, y_test)   # 0.7872
    # # SVM
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    # # mlp
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)  # 0.80076

    # print("get_features_by_wordbag_tfidf")
    # x_train, x_test, y_train, y_test=get_features_by_wordbag_tfidf()
    # # NB
    # do_nb_wordbag(x_train, x_test, y_train, y_test)  # 0.80268
    # # SVM
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    # mlp
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)  # 0.87268  max_feature=5000

    # print("get_features_by_tf")  # 词汇表
    # x_train, x_test, y_train, y_test=get_features_by_tf()
    # # RNN
    # do_rnn_wordbag(x_train, x_test, y_train, y_test)

    # print("get_features_by_doc2vec")
    # x_train, x_test, y_train, y_test = get_features_by_doc2vec()
    # print(x_train.shape, x_test.shape, y_train[0:2])
    print("get_features_by_word2vec")
    x_train, x_test, y_train, y_test = get_feature_by_word2vec('word2vec.bin')
    print(x_train.shape)
    # # NB
    # do_nb_doc2vec(x_train, x_test, y_train, y_test)    # word2vec 0.65764
    # # SVM
    do_svm_wordbag(x_train, x_test, y_train, y_test)  # 0.87712   word2vec 0.85312
    # mlp
    # do_dnn_wordbag(x_train, x_test, y_train, y_test)  # doc2vec 0.87184   word2vec 0.85864
    # rf
    do_rf_doc2vec(x_train, x_test, y_train, y_test)   # 0.71004    word2vec 0.7348
    # CNN

    do_cnn_doc2vec(x_train, x_test, y_train, y_test)  # 0.5172  word2vec val_acc: 0.6514

    # word2vec_demo()












