# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.layer import InputLayer,Layer,OutputLayer
from yjf.nn.network import NeuralNet
import time

if __name__ == '__main__':
    
    
    net = NeuralNet()
    
    inputlayer = InputLayer()
    inputlayer.setInputData(np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]]).T)    
    net.setInputLayer(inputlayer)
    
    net.addLayer(Layer(4,"ReLU"))    
    net.addLayer(Layer(4,"ReLU"))
    
    outputlayer =  OutputLayer(1,"sigmoid")
    outputlayer.setExpectedOutput(np.array([0,0,0,1,1]).reshape(1,5))    
    net.addLayer(outputlayer)
    
    print("begin to train ... " )
    tic = time.time()
    net.train()            
    toc = time.time()
    print(net.getOutput())
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict ... " )
    testlayer = InputLayer()
    testlayer.setInputData(np.array([[2,8,3], [555,666,777],[55,66,77]]).T)    
    net.setInputLayerForPredict(testlayer)
    net.forward()
    print(net.getOutput())
    