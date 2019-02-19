#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


import numpy as np


def nextArray(modeStr):
    next = np.zeros((len(modeStr), ), dtype='int32')
    next[0] = -1
    i = 0
    j = -1
    while i < len(modeStr)-1:
        if j == -1 or modeStr[i] == modeStr[j]:
            i += 1
            j += 1
            next[i] = j
        else:
            j = next[j]
    return next


def nextArray2(modeStr):
    next = np.zeros((len(modeStr), ), dtype='int32')
    next[0] = -1
    j = 0
    k = -1
    while j < len(modeStr)-1:
        if k == -1 or modeStr[j] == modeStr[k]:
            j += 1
            k += 1
            if modeStr[j] == modeStr[k]:
                next[j] = next[k]
            else:
                next[j] = k
        else:
            k = next[k]
    return next


def KMP(mainStr, modeStr, next):
    i = 0
    j = 0
    while i < len(mainStr) and j < len(modeStr):
        if j == -1 or mainStr[i] == modeStr[j]:
            i += 1
            j += 1
        else:
            j = next[j]
    if j == len(modeStr):
        return i-len(modeStr)
    else:
        return -1


if __name__ == '__main__':
    print(nextArray('ababc'))
    print(nextArray2('abcaacbbcbadaabcacbd'))
    modestr = 'ababc'
    next = nextArray2(modestr)
    print(KMP('abababc', modestr, next))
    print(KMP('ababaabcababc', modestr, next))
