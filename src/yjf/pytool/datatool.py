# -*- coding: UTF-8 -*-
'''
Created on 2020年6月26日

@author: yangjinfeng
'''
import numpy as np


def normalizerData(data):
    miu = np.mean(data,axis = 1,keepdims=True)
    sigma = np.mean(data * data,axis = 1,keepdims=True)
    return (data -miu)/sigma


def sigmoid(x):
    min = -50
    max = 50
    tempz = np.maximum(min,np.minimum(max,x))
    return 1 / (1 + np.exp(-tempz)) 



if __name__ == '__main__':
    data = np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]]).T
    print(sigmoid(data))
    newdata = normalizerData(data)
    print(newdata)
    