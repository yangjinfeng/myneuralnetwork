# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.wbvector import ParamVectorConverter
from yjf.nn.cfg import HyperParameter

class NeuralNet:


    def __init__(self,iters=100):
        '''
        Constructor
        '''
        self.iters = iters
        self.layers=[]
        self.mode = True  #True train; False test
        self.iterCounter = 0
        
    
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
            
    
    #True train; False test
    def setMode(self,train_test):
        self.mode = train_test
            
    
    def forward(self):
        if self.mode:
            self.iterCounter = self.iterCounter + 1
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
        训练样本的拟合结果
    '''
    def getPrediction(self):
        return self.layers[len(self.layers)-1].predict()
    
    '''
        每一次前向传播的损失+L2正则项
    '''
    def getLoss(self):
        loss = self.layers[len(self.layers)-1].loss()
        if(HyperParameter.L2_Reg):
            paramVec = ParamVectorConverter.toParamVector(self)
            L2loss = (HyperParameter.L2_lambd/(2 * self.dataSize) ) * (np.sum(paramVec * paramVec)) #L2正则化的损失
            loss = loss + L2loss
        return loss
    
    
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

    def train(self,logger = None):
        self.initialize()            
        for i in range(self.iters):
            self.forward()
            if logger is not None:
                logger.log()
            self.backward()
        
    def predict(self):    
        self.setMode(False)
        self.forward()