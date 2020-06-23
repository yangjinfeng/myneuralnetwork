# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np

class NeuralNet:


    def __init__(self):
        '''
        Constructor
        '''
        self.layers=[]
        
    
    def setInputLayer(self,inputLayer):    
        self.inputLayer = inputLayer
        self.dataSize = inputLayer.M

    def setInputLayerForPredict(self,inputLayer):    
        self.inputLayer = inputLayer
        self.dataSize = inputLayer.M
        self.layers[0].setPreLayer(self.inputLayer)
        
    
    def addLayer(self, layer):
        layerLen = len(self.layers)
        if(layerLen > 0):
            topLayer = self.layers[layerLen-1]
            topLayer.setNextLayer(layer)
            layer.setPreLayer(topLayer)
        self.layers.append(layer)
    
    def initialize(self):
        self.layers[0].setPreLayer(self.inputLayer)
        for layer in self.layers:
            layer.setDataSize(self.dataSize);
            layer.initialize()
            
    
    def forward(self):
        for layer in self.layers:
            layer.forward()
#         currentLayer = layer1
#         while(True):
#             currentLayer.forward()
#             if(not currentLayer.isOutputLayer()):
#                 currentLayer = currentLayer.nextLayer
#             else:
#                 break
    '''
            训练结束后，对B在训练样本上求平均，B是一个列向量（4,1）
             对预测来说，如果输入X是(3,1),W是（4,3），W×X就是（4,1）
    '''
    def avgB(self):
        for layer in self.layers:
            layer.B = (1/layer.M)* np.sum(layer.B, axis = 1, keepdims=True)
        


    def getOutput(self):
        return self.layers[len(self.layers)-1].A
    
    def backward(self):
        currentLayer = self.layers[len(self.layers)-1]
        while(True):
            if(not currentLayer.isInputLayer()):
                currentLayer.backward()
                currentLayer = currentLayer.preLayer
            else:
                break    


    def train(self):
        self.initialize()    
        for i in range(10000):
            self.forward()
            self.backward()
        self.avgB()
        
        