#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn.layers.recurrent import bidirectional_rnn
from tflearn.layers.recurrent import BasicLSTMCell
from tflearn.layers.estimator import regression
from tflearn.datasets import imdb
from tflearn.data_utils import pad_sequences, to_categorical


def single_lstm(train_x, train_y, test_x, test_y):
    train_x = pad_sequences(train_x, maxlen=100, value=0.)
    test_x = pad_sequences(test_x, maxlen=100, value=0.)

    train_y = to_categorical(train_y, nb_classes=2)
    test_y = to_categorical(test_y, nb_classes=2)

    input = input_data(shape=[None, 100], name='input')
    embedding = tflearn.embedding(input, input_dim=10000, output_dim=128)
    lstm1 = lstm(embedding, 128, dropout=0.8)
    fc = fully_connected(lstm1, 2, activation='softmax')
    net = regression(fc, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(train_x, train_y, n_epoch=5, validation_set=(test_x, test_y), show_metric=True,
              batch_size=32, run_id='single_lstm')


def bi_lstm(train_x, train_y, test_x, test_y):
    train_x = pad_sequences(train_x, maxlen=200, value=0.)
    test_x = pad_sequences(test_x, maxlen=200, value=0.)

    train_y = to_categorical(train_y, nb_classes=2)
    test_y = to_categorical(test_y, nb_classes=2)

    input = input_data(shape=[None, 200], name='input')
    embedding = tflearn.embedding(input, input_dim=20000, output_dim=128)
    bi_lstm1 = bidirectional_rnn(embedding, BasicLSTMCell(128), BasicLSTMCell(128))
    dr = tflearn.dropout(bi_lstm1, 0.5)

    fc = fully_connected(dr, 2, activation='softmax')
    net = tflearn.regression(fc, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="bi_logs/")
    model.fit(train_x, train_y, n_epoch=4, validation_set=(test_x, test_y),
              show_metric=True, batch_size=64, run_id='bi_lstm')

    print(test_x[0:2], test_y[0:2])
    print(model.predict(test_x[0].reshape((1, 200))))
    print(model.predict_label(test_x[0].reshape((1, 200))))
    print(model.predict(test_x[1].reshape((1, 200))))
    print(model.predict_label(test_x[1].reshape((1, 200))))


def shakespeare():
    import os
    from six.moves import urllib
    import pickle
    from tflearn.data_utils import textfile_to_semi_redundant_sequences, random_sequence_from_textfile
    path = 'shakespeare_input.txt'
    char_idx_file = 'char_idx.pickle'

    if not os.path.exists(path):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt", path)
    maxlen = 25
    char_idx = None
    if os.path.isfile(char_idx_file):
        print('Loading previous char_idx')
        char_idx = pickle.load(open(char_idx_file, 'rb'))
    X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                                          pre_defined_char_idx=char_idx)

    pickle.dump(char_idx, open(char_idx_file, 'wb'))

    g = tflearn.input_data([None, maxlen, len(char_idx)])
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512, return_seq=True)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.lstm(g, 512)
    g = tflearn.dropout(g, 0.5)
    g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
    g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                           learning_rate=0.001)

    m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path='model_shakespeare')

    for i in range(5):
        seed = random_sequence_from_textfile(path, maxlen)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='shakespeare')
        print("-- TESTING...")
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        # print(m.generate(10, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))


if __name__ == '__main__':
    train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
    train_x, train_y = train
    test_x, test_y = test
    # single_lstm(train_x, train_y, test_x, test_y)
    # bi_lstm(train_x, train_y, test_x, test_y)
    shakespeare()