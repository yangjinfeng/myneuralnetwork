# -*- coding: UTF-8 -*-
'''
Created on 2020年8月16日

@author: yangjinfeng
'''

import tensorflow as tf

def f1():
    x = tf.constant(3.0)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x * x
    dy_dx = g.gradient(y, x) # Will compute to 6.0
    print(dy_dx)

if __name__ == '__main__':
    f1()