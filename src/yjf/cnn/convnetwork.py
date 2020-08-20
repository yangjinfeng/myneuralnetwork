# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''

from yjf.cnn.layer import Layer,InputLayer,FCCLayer,ConvLayer,BinaryOutputLayer
from yjf.data.datapy import DataContainer
from yjf.cnn.datasetutil import loadData1
import numpy as np

class ConvNet(object):
    '''
    classdocs
    '''


    def __init__(self,iters=100):
        '''
        Constructor
        '''
        self.iters = iters
        self.mode = True  #True train; False test
        self.layers=[]
        self.inputLayer = None
        self.iterCounter = 0
        self.dataContainer = DataContainer()
    
    def setTrainingData(self,trainX,trainY):    
        self.dataContainer.setTrainingData(trainX, trainY, False)
        self.inputLayer = InputLayer()
        self.inputLayer.A1_shape = trainX[0].shape
        
    def setTestData(self,testX,testY):
        self.dataContainer.setTestData(testX, testY)

    #True train; False test
    def setMode(self,train_test):
        self.mode = train_test
    
    
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
#         layer.setLayerIndex(len(self.layers)) #便于调试

    def forward(self):
        if self.mode:
            self.iterCounter = self.iterCounter + 1

        for layer in self.layers:
            layer.forward()


    def backward(self):
        currentLayer = self.layers[len(self.layers)-1]
        while(True):
            if(not currentLayer.isInputLayer()):
                currentLayer.backward()
                currentLayer = currentLayer.preLayer
            else:
                break    
    
    def getLoss(self):
        loss = self.layers[len(self.layers)-1].loss()
        return loss
        
    def trainOne(self,logger=None):
        self.inputLayer.setInputData(self.dataContainer.trainingdata[0][DataContainer.training_x])
        self.layers[len(self.layers)-1].setExpectedOutput(self.dataContainer.trainingdata[0][DataContainer.training_y])
        for i in range(self.iters):
            self.forward()
#             self.getPrediction()
            if logger is not None:
                logger.log()
            self.backward()

    '''
        训练样本的拟合结果
    '''
    def getPrediction(self):
        return self.layers[len(self.layers)-1].predict()


if __name__ == '__main__':
    convnet = ConvNet()
    X1,Y1 = loadData1()
    convnet.setTrainingData(X1, np.array([Y1]))
    
    convnet.addLayer(ConvLayer({"size":3,"stride":1,"padding":"SAME","count":3},"ReLU"))  
    convnet.addLayer(ConvLayer({"size":5,"stride":1,"padding":"SAME","count":4},"ReLU")) 
    convnet.addLayer(FCCLayer(100,"ReLU")) 
    convnet.addLayer(BinaryOutputLayer(1,"sigmoid",0.5))
    convnet.trainOne()
    