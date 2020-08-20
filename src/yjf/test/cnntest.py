# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''
from yjf.cnn.layer import FCCLayer,ConvLayer,BinaryOutputLayer,PoolLayer
from yjf.cnn.datasetutil import loadData1
from yjf.cnn.convnetwork import ConvNet
from yjf.nn.eval import Evaluator
from yjf.nn.logger import MyLogger
import numpy as np

'''
keras代码
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
'''

if __name__ == '__main__':
    np.random.seed(2)
    convnet = ConvNet(500)
    X,Y = loadData1()
    convnet.setTrainingData(X, Y)
    
    convnet.addLayer(ConvLayer({"size":5,"stride":2,"padding":"VALID","count":3},"ReLU"))  
    convnet.addLayer(ConvLayer({"size":5,"stride":2,"padding":"VALID","count":6},"ReLU")) 
#     convnet.addLayer(PoolLayer({"method":"MAX","size":3,"stride":1,"padding":"SAME"}))
    convnet.addLayer(ConvLayer({"size":5,"stride":2,"padding":"VALID","count":12},"ReLU")) 
    convnet.addLayer(ConvLayer({"size":5,"stride":2,"padding":"VALID","count":24},"ReLU")) 
    convnet.addLayer(FCCLayer(2000,"ReLU")) 
    convnet.addLayer(FCCLayer(500,"ReLU"))
    convnet.addLayer(BinaryOutputLayer(1,"sigmoid",0.5))
    
    logger = MyLogger("cnn_loss.log")
    logger.setNetwork(convnet)
    convnet.trainOne(logger)
    logger.close()
    prd = convnet.getPrediction()
    eval = Evaluator(Y,prd[1])
    print(eval.eval().toMatrix())
