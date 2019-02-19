#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.datasets import imdb
from tflearn.data_utils import to_categorical, pad_sequences
import pickle
import numpy as np

# IMDB 是一个电影评论的数据库,
# path 是存储的路经，pkl 是 byte stream 格式，用这个格式在后面比较容易转换成 list 或者 tuple。
# n_words 为从数据库中取出来的词个数。

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1,)
train_x, train_y = train
test_x, test_y = test
# print(train_x[0:2], train_y[0:2])

# 转化为固定长度的向量，这里固定长度为100
train_x = pad_sequences(train_x, maxlen=100, value=0)
# print(train_x[0:2])
test_x = pad_sequences(test_x, maxlen=100, value=0)

# 二值化向量
train_y = to_categorical(train_y, nb_classes=2)  # [0]=>[1. 0.]
test_y = to_categorical(test_y, nb_classes=2)   # [1]=>[0. 1.]
# print(train_y[0:2])

# 构建卷积神经网络，使用1d卷积

input = input_data(shape=[None, 100], name='input')
embedding = tflearn.embedding(input, input_dim=10000, output_dim=128)
branch1 = conv_1d(embedding, 128, 3, padding='valid', activation='relu', regularizer='L2')
branch2 = conv_1d(embedding, 128, 4, padding='valid', activation='relu', regularizer='L2')
branch3 = conv_1d(embedding, 128, 5, padding='valid', activation='relu', regularizer='L2')
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
"""
训练开始
"""
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='logs')
model.fit(train_x, train_y, n_epoch=1, shuffle=True, validation_set=(test_x, test_y), show_metric=True, batch_size=32)
"""
模型保存
"""
model.save("cnn_text.model")



