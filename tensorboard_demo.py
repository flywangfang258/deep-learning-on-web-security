#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# import tensorflow as tf
# with tf.name_scope('graph') as scope:
#      matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
#      matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
#      product = tf.matmul(matrix1, matrix2,name='product')
# sess = tf.Session()
# writer = tf.summary.FileWriter("C:\\logs\\test", sess.graph)
# init = tf.global_variables_initializer()
# sess.run(init)


import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
sess.run(e)

writer = tf.summary.FileWriter("F:/tensorflowe/graph", tf.get_default_graph())
writer.close()
