# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''
from yjf.cnn.layer import FCCLayer,ConvLayer,BinaryOutputLayer
from yjf.cnn.datasetutil import loadData1
from yjf.cnn.convnetwork import ConvNet
from yjf.nn.eval import Evaluator

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
    convnet = ConvNet()
    X,Y = loadData1()
    convnet.setTrainingData(X, Y)
    
    convnet.addLayer(ConvLayer({"size":3,"stride":1,"padding":"SAME","count":3},"ReLU"))  
    convnet.addLayer(ConvLayer({"size":5,"stride":1,"padding":"SAME","count":4},"ReLU")) 
    convnet.addLayer(FCCLayer(100,"ReLU")) 
    convnet.addLayer(BinaryOutputLayer(1,"sigmoid",0.5))
    convnet.trainOne()
    prd = convnet.getPrediction()
    eval = Evaluator(Y,prd)
    print(eval.eval().precision())
