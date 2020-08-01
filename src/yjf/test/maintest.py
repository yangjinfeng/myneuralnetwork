# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.layer import InputLayer,Layer,BinaryOutputLayer,MultiOutputLayer
from yjf.nn.network import NeuralNet
from yjf.data.datagen import DataGenerator
from yjf.nn.eval import Evaluator
from yjf.nn.logger import MyLogger
import time



def testOne():
    np.random.seed(1)
    net = NeuralNet(5000)
    
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadClassificationDataset()
    net.setMiniBatch(False)
    net.setTrainingData(Tr_x, Tr_y)
    
    
    net.addLayer(Layer(40,"ReLU",0.9))   
#     net.addLayer(Layer(40,"ReLU",0.9)) 
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"sigmoid",1))
    net.addLayer(Layer(20,"ReLU",0.9))     
    net.addLayer(BinaryOutputLayer(1,"sigmoid",1,0.5))
    
    
    logger = MyLogger("loss.log")
    logger.setNetwork(net)
    print("begin to train ... " )
    tic = time.time()
    net.train()
    logger.close()            
    
    toc = time.time()
#     fitting = net.getPrediction()
#     eval1 = Evaluator(Tr_y,fitting)
#     print(eval1.eval())
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict training data... " )
    net.setTestData(Tr_x, Tr_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(Tr_y,prd)
    print(eval2.eval())

    
    print("begin to predict test data... " )
    net.setTestData(T_x, T_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(T_y,prd)
    print(eval2.eval())


def testBatch():
    np.random.seed(1)
    net = NeuralNet(5000)
    
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadClassificationDataset()
    net.setMiniBatch(True)
    net.setTrainingData(Tr_x, Tr_y)
    
    
    net.addLayer(Layer(40,"ReLU",0.9))   
    net.addLayer(Layer(40,"ReLU",0.9)) 
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"ReLU",1))
#     net.addLayer(Layer(4,"sigmoid",1))
    net.addLayer(Layer(20,"ReLU",0.9))     
    net.addLayer(BinaryOutputLayer(1,"sigmoid",1,0.5))
    
    
    logger = MyLogger("loss.log")
    logger.setNetwork(net)
    print("begin to train ... " )
    tic = time.time()
    net.train()
    logger.close()            
    
    toc = time.time()
#     fitting = net.getPrediction()
#     eval1 = Evaluator(Tr_y,fitting)
#     print(eval1.eval())
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict training data... " )
    net.setTestData(Tr_x, Tr_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(Tr_y,prd)
    print(eval2.eval())

    
    print("begin to predict test data... " )
    net.setTestData(T_x, T_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(T_y,prd)
    print(eval2.eval())
    
def testNClassesBatch():
    np.random.seed(1)
    net = NeuralNet(1000)
    
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadNClassificationDataset2(4,5000,1000)
    net.setMiniBatch(True)
    net.setTrainingData(Tr_x, Tr_y)    
    
    net.addLayer(Layer(20,"ReLU",0.9))   
    net.addLayer(Layer(20,"ReLU",0.9)) 
    net.addLayer(Layer(20,"ReLU",0.9))
    net.addLayer(Layer(20,"ReLU",0.9))     
    net.addLayer(MultiOutputLayer(4,"softmax",1))
    
    
    print("begin to train ... " )
    tic = time.time()
    net.train()
    toc = time.time()
    print("time lasted: " + str(1000*(toc-tic)))
    
    print("begin to predict training data... " )
    net.setTestData(Tr_x, Tr_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(Tr_y,prd)
    print(eval2.evalN()) #0.9148

    
    print("begin to predict test data... " )
    net.setTestData(T_x, T_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(T_y,prd)
    print(eval2.evalN()) #0.315

if __name__ == '__main__':
    print("-----------trainOne------------------")
#     testOne()
    print("-----------trainBatch------------------")
#     testBatch()
    testNClassesBatch()
    
    