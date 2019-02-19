#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# -*- coding:utf-8 -*-
import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

char_idx_file = 'char_idx_xss.pkl'
maxlen = 25
char_idx = None
xss_data_file="../data/aiscanner/xss.txt"

def get_login_pages(keywords):
    from sklearn.datasets import fetch_20newsgroups
    import gensim
    import re
    """
    newsgroups_train = fetch_20newsgroups(subset='train')
    for  news in newsgroups_train.target_names:
        print news

    alt.atheism
    comp.graphics
    comp.os.ms-windows.misc
    comp.sys.ibm.pc.hardware
    comp.sys.mac.hardware
    comp.windows.x
    misc.forsale
    rec.autos
    rec.motorcycles
    rec.sport.baseball
    rec.sport.hockey
    sci.crypt
    sci.electronics
    sci.med
    sci.space
    soc.religion.christian
    talk.politics.guns
    talk.politics.mideast
    talk.politics.misc
    talk.religion.misc
    """
    #cats = ['sci.crypt']
    #newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    newsgroups=[]
    newsgroups.append(newsgroups_train.data)
    newsgroups.append(newsgroups_test.data)
    #newsgroups_train = fetch_20newsgroups()
    #print len(newsgroups_train.data)
    print(newsgroups_train.data)
    sentences=[re.findall("[a-z\-]+",s.lower()) for s in newsgroups_train.data]
    #sentences = [s.lower().split() for s in newsgroups_train.data]
    #print sentences

    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=1, workers=4,iter=20)

    #print len(sentences)

    for key in keywords:
        print("[%s] most_similar:" % key)
        results=model.most_similar(positive=[key], topn=10)
        for i in results:
            print(i)

def get_login_pages_imdb(keywords):

    import gensim
    import re
    from tflearn.datasets import imdb

    train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                    valid_portion=0.1)

    trainX, trainY = train
    sentences=trainX
    print(len(sentences))
    print(sentences)

    model = gensim.models.Word2Vec(sentences, size=200, window=3, min_count=1, workers=4,iter=50)



    for key in keywords:
        print("[%s] most_similar:" % key)
        results=model.most_similar(positive=[key], topn=10)
        for i in results:
            print(i)

if __name__ == "__main__":
    print("Hello ai scanner poc")
    get_login_pages(["user","password","email","name"])
    #get_login_pages_imdb(["user", "password", "email", "name"])