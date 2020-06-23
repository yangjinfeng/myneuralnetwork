# -*- coding: UTF-8 -*-
'''
Created on 2020年6月22日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.activate import factory

class Layer:
    def __init__(self,N,actvaname):
        self.N = N     #当前层
        self.alpha = 0.01
        self.actva = factory(actvaname)

        

    def setDataSize(self,dataSize):
        self.M = dataSize
        
    def setPreLayer(self,layer):
        self.preLayer = layer
        
    def setNextLayer(self,layer):
        self.nextLayer = layer

    def initialize(self):
        self.W = np.random.randn(self.N,self.preLayer.N)*0.01;
#         self.B = np.random.randn(self.N,self.M)*0.001;
        self.B = np.zeros((self.N,self.M),dtype=float);
        


        
    def forward(self):
        self.Z = np.matmul(self.W, self.preLayer.A) + self.B
#         self.A = self.activation(self.Z);
        self.A = self.actva.activate(self);
        
        
  
    def backward(self):
        if(self.isOutputLayer()):
#             self.dA = -self.Y/self.A + (1-self.Y)/(1-self.A) 
            self.dA = self.deriveLoss()
        else:
            self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ)
        
        self.dZ =  self.dA * self.actva.derivative(self)
        self.dW = (1/self.M)*np.matmul(self.dZ, self.preLayer.A.T)
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
        self.W = self.W - self.alpha * self.dW
        self.B = self.B - self.alpha * self.dB

    

    def isOutputLayer(self):
        return False
#         return not hasattr(self,'nextLayer')

    def isInputLayer(self):
        return False
#         return not hasattr(self,'preLayer')

    
    
    
class InputLayer(Layer):
    
    def __init__(self):
        pass
    
    
    def setInputData(self,data):
#         self.A = np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]]).T
#         self.Y = np.array([0,0,0,1,1]).reshape(1,5)
        self.A = data
        self.N = data.shape[0]
        self.M = data.shape[1]

    def isInputLayer(self):
        return True
   
   
   
class OutputLayer(Layer):
    
    
    '''
    a=y^=σ(z)
    Logistic regression: loss(a,y) = -( ylog(a) + (1-y)log(1-a)) 
    da = derivative(loss(a,y)) = -y/a + (1-y)/(1-a)
          输出层的损失函数对A求导，计算出dA即可，启动反向传播
    '''
    def deriveLoss(self):
        self.dA = -self.Y/self.A + (1-self.Y)/(1-self.A)
        return self.dA
    
    def setExpectedOutput(self,output):
#         self.Y = np.array([0,0,0,1,1]).reshape(1,5)
        self.Y = output

    def isOutputLayer(self):
        return True
  