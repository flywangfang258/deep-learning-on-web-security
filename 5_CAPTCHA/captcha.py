#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import tflearn
from tflearn.data_utils import to_categorical

def load_data():
    import tflearn.datasets.mnist as mnist
    x_train, y_train, x_test, y_test = mnist.load_data(one_hot=False)
    return x_train, y_train, x_test, y_test


def knn_1d(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=15)
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('knn_acc', metrics.accuracy_score(y_test, y_pred))


def svm_1d(x_train, y_train, x_test, y_test):
    clf = SVC(kernel='rbf', decision_function_shape='ovo')
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('svm_acc', metrics.accuracy_score(y_test, y_pred))


def mlp_2d(x_train, y_train, x_test, y_test):

    print(x_train.shape)
    input_layer = input_data([None, 784])
    dense1 = fully_connected(input_layer, 64, activation='tanh',
                             regularizer='L2', weight_decay=0.001)
    dropout1 = dropout(dense1, 0.8)
    dense2 = fully_connected(dropout1, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
    dropout2 = dropout(dense2, 0.8)
    softmax = fully_connected(dropout2, 10, activation='softmax')
    # Regression using SGD with learning rate decay and Top-3 accuracy
    # 反向传播使用SGD随机梯度下降法，学习率递减
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    # 定义，真实结果在预测结果前3中就算正确
    top_k = tflearn.metrics.Top_k(3)
    # 定义回归策略
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(x_train, y_train, n_epoch=20, validation_set=(x_test, y_test),
              show_metric=True, run_id="mlp_model")


def do_cnn_2d(X, Y, testX, testY ):
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    # Building convolutional network
    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': Y}, n_epoch=5, validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='cnn_mnist')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    # print(y_train[0:2])
    # knn_1d(x_train, y_train, x_test, y_test)
    # svm_1d(x_train, y_train, x_test, y_test)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # mlp_2d(x_train, y_train, x_test, y_test)

    do_cnn_2d(x_train, y_train, x_test, y_test)