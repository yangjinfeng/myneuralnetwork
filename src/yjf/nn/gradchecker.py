# -*- coding: UTF-8 -*-
'''
Created on 2020年6月27日

@author: yangjinfeng
'''
from yjf.nn.wbvector import ParamVectorConverter
import numpy as np
from yjf.nn.layer import InputLayer,Layer,OutputLayer
from yjf.nn.network import NeuralNet
from yjf.data.datagen import DataGenerator


class Checker(object):
    '''
    classdocs
    '''


    def __init__(self, network):
        '''
        Constructor
        '''
        self.network = network
        self.epsilon = 1e-7
        
    
    def getGrad(self):
        self.network.forward()
        self.network.backward()
        return ParamVectorConverter.toGradVector(self.network)
    
    
    def getApproxGrad(self):
        paramVec = ParamVectorConverter.toParamVector(self.network)
        p = np.copy(paramVec)
        paramNum = len(paramVec)
        approxGrad = np.zeros(paramNum,dtype=float);
        loss_plus = np.zeros(paramNum,dtype=float);
        loss_minus = np.zeros(paramNum,dtype=float);
        for i in range(paramNum):
#             thetaVec = np.copy(paramVec)
            origin = paramVec[i]
            
            paramVec[i] = origin + self.epsilon            
            lss = ParamVectorConverter.computeLoss(self.network, paramVec)            
            loss_plus[i] = lss[0]
            
            paramVec[i] = origin - self.epsilon
            lss = ParamVectorConverter.computeLoss(self.network, paramVec)            
            loss_minus[i] = lss[0]
            
            paramVec[i] = origin #还原
            
            approxGrad[i] = (loss_plus[i] - loss_minus[i]) / (2.0 * self.epsilon)            
#             approxGrad[i] = (loss_plus[i] - loss_minus[i] + 2*origin*self.epsilon*0.01) / (2.0 * self.epsilon)
        
        ParamVectorConverter.fillNet(self.network, paramVec) #还原最开始的网络参数
            
        return approxGrad
    
    
    def checkGrad(self):
        approxGrad = self.getApproxGrad()
        grad = self.getGrad()        
        numerator = np.linalg.norm(grad - approxGrad)      
        denominator = np.linalg.norm(grad) + np.linalg.norm(approxGrad)  
        difference = numerator / denominator
                
        print(approxGrad)
        print(grad)
        
        return difference
            



if __name__ == '__main__':
    net = NeuralNet()
    
    inputlayer = InputLayer()
#     inputlayer.setInputData(np.array([[1,2,3]]).T)    
    
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadDataset()
    inputlayer.setInputData(Tr_x[:,0].reshape(2,1))
    
    net.setInputLayer(inputlayer)
    
    net.addLayer(Layer(4,"ReLU",1))    
    net.addLayer(Layer(4,"ReLU",1)) #这一层设为dropout的话，output层的Z会非常大
    
    outputlayer =  OutputLayer(1,"sigmoid",1)
#     outputlayer.setExpectedOutput(np.array([0]).reshape(1,1))   
    outputlayer.setExpectedOutput(Tr_y[:,0].reshape(1,1)) 
    net.addLayer(outputlayer)
    net.initialize()
    
    #考虑正则化项的损失后，梯度检验的结果数量级在e-4
    checker = Checker(net)
    diff = checker.checkGrad()
    print(diff)
    
    
    
    
              