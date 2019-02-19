#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def load_dataset():
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets('MNIST_data/', one_hot=True)
    x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    x_train = np.array(x_train).reshape([-1, 28, 28, 1])
    print(x_train.shape)  # (55000, 28, 28, 1)
    x_test = np.array(x_test).reshape([-1, 28, 28, 1])
    print(x_test.shape)   # (10000, 28, 28, 1)

    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': x_train}, {'target': y_train}, n_epoch=20,
              validation_set=({'input': x_test}, {'target': y_test}),
              snapshot_step=100, show_metric=True, run_id='cnn_demo')


