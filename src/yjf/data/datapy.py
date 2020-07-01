# -*- coding: UTF-8 -*-
'''
Created on 2020年7月1日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.cfg import HyperParameter

class DataContainer(object):
    
    training_x = "training_x"
    training_y = "training_y"
    test_x = "test_x"
    test_y = "test_y "
    def __init__(self):
        self.trainingdata = [] #每一个元素是一个字典，x和y
        self.testdata = {} #字典，x和y
        self.predictdata = None
    
    def setTrainingData(self,traindataX,traindataY,miniBatch):
        if miniBatch:
            xbatch = DataContainer.batchData(traindataX)
            ybatch = DataContainer.batchData(traindataY)
            for i in range(len(xbatch)):
                xydict = {}
                xydict[DataContainer.training_x] = xbatch[i]
                xydict[DataContainer.training_y] = ybatch[i]
                self.trainingdata.append(xydict)
        else:
            xydict = {}
            xydict[DataContainer.training_x] = traindataX
            xydict[DataContainer.training_y] = traindataY
            self.trainingdata.append(xydict)
    
    def setTestData(self,testdataX,testdataY):
        self.testdata[DataContainer.test_x] = testdataX
        self.testdata[DataContainer.test_y] = testdataY
    
    def setPredictData(self,predictdata):
        self.predictdata = predictdata
    
    @staticmethod
    def batchData(data):
        batches = []
        size = HyperParameter.batch_size
        batch_num = int(data.shape[1] / size)
        count = 0
        for i in range(batch_num):
            batches.append(data[:,count:count + size])
            count = count + size
        if data.shape[1] > count:
            batches.append(data[:,count:data.shape[1]])
        return batches


if __name__ == '__main__':
    data= np.random.rand(3,20)
    batches = DataContainer.batchData(data)
    for i in range(len(batches)):
        b = batches[i]
        print(b.shape)
        print(b)