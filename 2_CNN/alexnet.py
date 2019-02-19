#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# AlexNet 一共有8个层组成，其中5个卷基层，3个全连接层

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn

import tflearn.datasets.oxflower17 as oxflower17

def alexnet():
    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    input = input_data([None, 227, 227, 3], name='input')
    conv1 = conv_2d(input, 96, 11, strides=4, activation='relu')
    pool1 = max_pool_2d(conv1, 3, strides=2)
    conv2 = conv_2d(pool1, 256, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 3, strides=2)
    conv3 = conv_2d(pool2, 384, 3, activation='relu')
    conv4 = conv_2d(conv3, 384, 3, activation='relu')
    conv5 = conv_2d(conv4, 256, 3, activation='relu')
    pool3 = max_pool_2d(conv5, 3, strides=2)
    lrn = local_response_normalization(pool3)
    fc1 = fully_connected(lrn, 4096, activation='tanh')
    dr1 = dropout(fc1, 0.5)
    fc2 = fully_connected(dr1, 4096, activation='tanh')
    dr2 = dropout(fc2, 0.5)
    fc3 = fully_connected(dr2, 17, activation='softmax')

    # 声明优化算法、损失函数、学习率等
    network = regression(fc3, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=10, tensorboard_verbose=2, tensorboard_dir='logs')
    # n_epoch=10表示整个训练数据集将会用10遍，
    # batch_size=16表示一次用16个数据计算参数的更新
    model.fit(X, Y, n_epoch=8, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet')


if __name__ == '__main__':
    alexnet()