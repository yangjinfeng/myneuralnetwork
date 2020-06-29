# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np

class NeuralNet:


    def __init__(self,iters=100):
        '''
        Constructor
        '''
        self.iters = iters
        self.layers=[]
        self.mode = True
        
    
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
        index = 0
        self.inputLayer.setLayerIndex(index)
        for layer in self.layers:
            index = index + 1
            layer.setLayerIndex(index) #便于调试
            layer.setDataSize(self.dataSize);
            layer.initialize()
    
    def copy(self):
        newnet = NeuralNet()
        for layer in self.layers:
            newlayer = layer.copy()
            newnet.addLayer(newlayer)
        return newnet
            
    
    def setMode(self,train_test):
        #True train; False test
        self.mode = train_test
            
    
    def forward(self):
        for layer in self.layers:
            layer.forward(self.mode)
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
        

    '''
        训练样本的拟合结果
    '''
    def getFittingResult(self):
        return self.layers[len(self.layers)-1].A
    
    '''
        每一个样本的最终损失
    '''
    def getLoss(self):
        return self.layers[len(self.layers)-1].loss()
    
    
    def backward(self):
        currentLayer = self.layers[len(self.layers)-1]
        while(True):
            if(not currentLayer.isInputLayer()):
                currentLayer.backward()
                currentLayer = currentLayer.preLayer
            else:
                break    


    def printLayers(self,file):
        for layer in self.layers:
            layer.outputInfo(file)

    def train(self):
        self.initialize()    
        file = open("network.txt","w")
        for i in range(self.iters):
            self.forward()
#             file.write("第  "+str(i)+" 轮: \n")
#             self.printLayers(file)
            self.backward()
        file.close()
        self.avgB()
        
        