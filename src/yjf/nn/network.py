# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
# from yjf.nn.wbvector import ParamVectorConverter
# from yjf.nn.cfg import HyperParameter
from yjf.data.datapy import DataContainer
from yjf.nn.layer import InputLayer
from yjf.nn.env import globalEnv

class NeuralNet:


    def __init__(self,iters=100):
        self.iters = iters
        self.layers=[]
        self.mode = True  #True train; False test
        self.isMiniBatch = False
        self.iterCounter = 0
        self.inputLayer = None
        self.dataContainer = DataContainer()
        
        
    def setTrainingData(self,trainX,trainY):    
        self.dataContainer.setTrainingData(trainX, trainY, self.isMiniBatch)
        self.inputLayer = InputLayer()
        self.inputLayer.N = trainX.shape[0]
        
    def setTestData(self,testX,testY):
        self.dataContainer.setTestData(testX, testY)

        
    
    def addLayer(self, layer):
        layerLen = len(self.layers)
        layer.setNetWork(self) #各层可以通过network共享一些变量        
        if(layerLen > 0):
            topLayer = self.layers[layerLen-1]
            topLayer.setNextLayer(layer)
            layer.setPreLayer(topLayer)
        else:
            layer.setPreLayer(self.inputLayer)  #layer是第一层
        self.layers.append(layer)
        layer.initialize()
        layer.setLayerIndex(len(self.layers)) #便于调试
        
    
#     '''
#            组装网络， 初始化各层的参数
#     '''
#     def initialize(self):
#         self.inputLayer = InputLayer()
#         self.layers[0].setPreLayer(self.inputLayer)
#         index = 0
#         self.inputLayer.setLayerIndex(index)
#         for layer in self.layers:
#             index = index + 1
#             layer.setLayerIndex(index) #便于调试
# #             layer.setDataSize(self.dataSize);
#             layer.initialize()
    
    def copy(self):
        newnet = NeuralNet()
        for layer in self.layers:
            newlayer = layer.copy()
            newnet.addLayer(newlayer)
        return newnet
            
    
    #True train; False test
    def setMode(self,train_test):
        self.mode = train_test
            
    
    def setMiniBatch(self,isMiniBatch):
        self.isMiniBatch = isMiniBatch
    
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
        每一次前向传播的损失+L2正则项
    '''
    def getLoss(self):
        loss = self.layers[len(self.layers)-1].loss()
        return loss

    
    
    def backward(self):
        currentLayer = self.layers[len(self.layers)-1]
        while(True):
            if(not currentLayer.isInputLayer()):
                if self.isMiniBatch:
#                     currentLayer.backward_mini_batch(t)
                    currentLayer.backward_batchnorm()
                else:
                    currentLayer.backward()
                currentLayer = currentLayer.preLayer
            else:
                break    


    def printLayers(self,file):
        for layer in self.layers:
            layer.outputInfo(file)

    '''
            训练整个数据集
    '''
    def trainOne(self):
        self.inputLayer.setInputData(self.dataContainer.trainingdata[0][DataContainer.training_x])
        self.layers[len(self.layers)-1].setExpectedOutput(self.dataContainer.trainingdata[0][DataContainer.training_y])
        for i in range(self.iters):
            self.forward()
            self.backward()

    '''
        小批次训练
    '''
    def trainMiniBatch(self):
        for i in range(self.iters):
            globalEnv.currentEpoch= i+1
            trainingdata = self.dataContainer.trainingdata
            for t in range(len(trainingdata)):
                globalEnv.currentBatch = t+1
                self.inputLayer.setInputData(trainingdata[t][DataContainer.training_x])
                self.layers[len(self.layers)-1].setExpectedOutput(trainingdata[t][DataContainer.training_y])
                self.forward()
                self.backward()
                



    '''
          模型训练
    '''
    def train(self):
        if self.isMiniBatch:
            self.trainMiniBatch()
        else:
            self.trainOne()

#     def train(self,logger = None):
#         self.initialize()            
#         for i in range(self.iters):
#             self.forward()
#             if logger is not None:
#                 logger.log()
#             self.backward()
        
    def test(self):    
        self.setMode(False)
        self.inputLayer.setInputData(self.dataContainer.testdata[DataContainer.test_x])
        self.layers[len(self.layers)-1].setExpectedOutput(self.dataContainer.testdata[DataContainer.test_y])
        self.forward()
        
    '''
        训练样本的拟合结果
    '''
    def getPrediction(self):
        return self.layers[len(self.layers)-1].predict()
    
        