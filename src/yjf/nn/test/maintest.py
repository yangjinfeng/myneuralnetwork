# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.layer import InputLayer,Layer,OutputLayer
from yjf.nn.network import NeuralNet
from yjf.data.datagen import DataGenerator
import time

def test1():
    net = NeuralNet()
    
    inputlayer = InputLayer()
    inputlayer.setInputData(np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]]).T)
    net.setInputLayer(inputlayer)
    
    net.addLayer(Layer(4,"ReLU",1))    
    net.addLayer(Layer(4,"ReLU",1)) #这一层设为dropout的话，output层的Z会非常大
    
    outputlayer =  OutputLayer(1,"sigmoid",1)
    outputlayer.setExpectedOutput(np.array([0,0,0,1,1]).reshape(1,5))    
    net.addLayer(outputlayer)
    
    print("begin to train ... " )
    tic = time.time()
    net.train()            
    toc = time.time()
    print(net.getOutput())
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict ... " )
#     testlayer = InputLayer()
    inputlayer.setInputData(np.array([[2,8,3], [555,666,777],[550,66,77]]).T)
    net.setInputLayerForPredict(inputlayer)
    net.setMode(False)
    net.forward()
    print(net.getOutput())


def test2():
    net = NeuralNet(1000)
    np.random.seed(1)
    
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadDataset()
    
    inputlayer = InputLayer()
    inputlayer.setInputData(Tr_x)    
    net.setInputLayer(inputlayer)
    
    net.addLayer(Layer(4,"sigmoid",1))    
    net.addLayer(Layer(4,"sigmoid",1)) #这一层设为dropout的话，output层的Z会非常大
    
    outputlayer =  OutputLayer(1,"sigmoid",1)
    outputlayer.setExpectedOutput(Tr_y)    
    net.addLayer(outputlayer)
    
    print("begin to train ... " )
    tic = time.time()
    net.train()            
    toc = time.time()
    print(net.getOutput())
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict ... " )
#     testlayer = InputLayer()
    inputlayer.setInputData(T_x)
    net.setInputLayerForPredict(inputlayer)
    net.setMode(False)
    net.forward()
    print(net.getOutput())

if __name__ == '__main__':
    
#     test1()
    test2()
    
    
    