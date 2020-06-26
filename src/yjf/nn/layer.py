# -*- coding: UTF-8 -*-
'''
Created on 2020年6月22日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.activate import factory

class Layer:
    def __init__(self,N,actvaname,keep_prob):
        self.N = N     #当前层
        self.alpha = 0.01 #learning rate
        self.actva = factory(actvaname)
        self.keep_prob = keep_prob #dropout
        self.lambd = 0.01 #L2的参数λ
        self.L2_Reg = True

        

    def setDataSize(self,dataSize):
        self.M = dataSize
        
    def setPreLayer(self,layer):
        self.preLayer = layer
        
    def setNextLayer(self,layer):
        self.nextLayer = layer

    def initialize(self):
        self.W = np.random.randn(self.N,self.preLayer.N) * 0.01;
#         self.W = np.random.randn(self.N,self.preLayer.N) * np.sqrt(2/(self.preLayer.N));
#         self.B = np.random.randn(self.N,self.M)*0.001;
        self.B = np.zeros((self.N,self.M),dtype=float);
        

        
        
    def forward(self,dropout):
        
        self.Z = np.matmul(self.W, self.preLayer.A) + self.B
#         self.A = self.activation(self.Z);
        self.A = self.actva.activate(self);
        #do dropout
        if (dropout):
            probMatrix = np.random.rand(self.A.shape[0],self.A.shape[1]) < self.keep_prob        
            self.A = np.multiply(self.A, probMatrix) / self.keep_prob
            
        
  
    def backward(self):
        if(self.isOutputLayer()):
            #尽量不做除法
#             self.dA = -self.Y/self.A + (1-self.Y)/(1-self.A) 
#             self.dA = self.deriveLoss()
            self.dZ = self.deriveLossByZ()
        else:
            self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ)
            self.dZ =  self.dA * self.actva.derivative(self)        
        
#         self.dZ =  self.dA * self.actva.derivative(self)
        self.dW = (1/self.M)*np.matmul(self.dZ, self.preLayer.A.T)
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
        
        #L2 regularization
        weight_decay = 1
        if(self.L2_Reg):
            weight_decay = 1 - self.alpha * self.lambd / self.M
        self.W = self.W * weight_decay - self.alpha * self.dW
        self.B = self.B * weight_decay - self.alpha * self.dB

    

    def isOutputLayer(self):
        return False
#         return not hasattr(self,'nextLayer')

    def isInputLayer(self):
        return False
#         return not hasattr(self,'preLayer')

    
    def outputInfo(self,outer):
        if(hasattr(self,'W')):
            outer.write("W: \n")
            outer.write(str(self.W)+"\n")
#         if(hasattr(self,'dW')):
#             outer.write("dW: \n")
#             outer.write(str(self.dW)+"\n")
        if(hasattr(self,'B')):
            outer.write("B: \n")
            outer.write(str(self.B)+"\n")
#         if(hasattr(self,'dB')):
#             outer.write("dB: \n")
#             outer.write(str(self.dB)+"\n")
        if(hasattr(self,'Z')):
            outer.write("Z: \n")
            outer.write(str(self.Z)+"\n")
#         if(hasattr(self,'dZ')):
#             outer.write("dZ: \n")
#             outer.write(str(self.dZ)+"\n")
        if(hasattr(self,'A')):
            outer.write("A: \n")
            outer.write(str(self.A)+"\n")
#         if(hasattr(self,'dA')):
#             outer.write("dA: \n")
#             outer.write(str(self.dA)+"\n")
           
    
    
class InputLayer(Layer):
    
    def __init__(self):
        pass
    
    def normalizerData(self,data):
        if(not (hasattr(self,'miu') and hasattr(self,'sigma'))):
            self.miu = np.mean(data,axis = 1,keepdims=True)
            self.sigma = np.mean(data * data,axis = 1,keepdims=True)
        return (data -self.miu)/self.sigma
    
    def setInputData(self,data):
#         self.A = np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]]).T
#         self.Y = np.array([0,0,0,1,1]).reshape(1,5)
        self.A = self.normalizerData(data)
#         self.A = data
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

    '''
    dLoss/dA * dA/dZ
    '''
    def deriveLossByZ(self):
        return self.A - self.Y

    
    def setExpectedOutput(self,output):
#         self.Y = np.array([0,0,0,1,1]).reshape(1,5)
        self.Y = output

    def isOutputLayer(self):
        return True
  