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
        self.actva = factory(actvaname)
        self.keep_prob = keep_prob #dropout
        
        self.alpha = 0.01 #learning rate
        self.lambd = 0.01 #L2的参数λ
        self.L2_Reg = True
        #初始化成员
        self.M = None
        self.W = None
        self.dW = None
        self.B = None
        self.dB = None
        self.A = None
        self.dA = None
        self.Z = None
        self.dZ = None
        self.preLayer = None
        self.nextLayer = None
    
    def setLayerIndex(self,index):
        self.index = index

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
#         self.B = np.zeros((self.N,self.M),dtype=float);
        self.B = np.random.randn(self.N,self.M) * 0.001
        
    def copy(self):
        newlayer = Layer(self.N,"",self.keep_prob)
        newlayer.W = self.W
        newlayer.B = self.B
        newlayer.actva = self.actva
        return newlayer
        
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
#             self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ) #这个bug找了好久
            self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)
            self.dZ =  self.dA * self.actva.derivative(self)        
        
#         self.dZ =  self.dA * self.actva.derivative(self)
        self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
        
        #L2 regularization
        weight_decay = 1
        if(self.L2_Reg):
            weight_decay = 1 - self.alpha * self.lambd / self.M
        self.W0 = np.copy(self.W) #给前一层计算梯度使用
        self.W = self.W * weight_decay - self.alpha * self.dW
        self.B = self.B * weight_decay - self.alpha * self.dB

    

    def isOutputLayer(self):
        return False
#         return not hasattr(self,'nextLayer')

    def isInputLayer(self):
        return False
#         return not hasattr(self,'preLayer')

    
    def outputInfo(self,outer):
        outer.write("第  "+str(self.index)+" 层: \n")
        if(self.W is not None):
            outer.write("W: \n")
            outer.write(str(self.W)+"\n")
        if(self.dW is not None):
            outer.write("dW: \n")
            outer.write(str(self.dW)+"\n")
        if(self.B is not None):
            outer.write("B: \n")
            outer.write(str(self.B)+"\n")
        if(self.dB is not None):
            outer.write("dB: \n")
            outer.write(str(self.dB)+"\n")
        if(self.Z is not None):
            outer.write("Z: \n")
            outer.write(str(self.Z)+"\n")
        if(self.dZ is not None):
            outer.write("dZ: \n")
            outer.write(str(self.dZ)+"\n")
        if(self.A is not None):
            outer.write("A: \n")
            outer.write(str(self.A)+"\n")
        if(self.dA is not None):
            outer.write("dA: \n")
            outer.write(str(self.dA)+"\n")
           
    
    
class InputLayer(Layer):
    
    def __init__(self):
        pass
    
    def normalizerData(self,data):
        if(not (hasattr(self,'miu') and hasattr(self,'sigma'))):
            self.miu = np.mean(data,axis = 1,keepdims=True)
            self.sigma = np.mean(data * data,axis = 1,keepdims=True)
        return (data -self.miu)/self.sigma
    
    def setInputData(self,data):
        self.A = data
#         self.A = self.normalizerData(data)
        self.N = data.shape[0]
        self.M = data.shape[1]

    def isInputLayer(self):
        return True
   
   
   
class OutputLayer(Layer):
    
    
    
    def loss(self):
        return np.multiply(-np.log(self.A), self.Y) + np.multiply(-np.log(1 - self.A), 1 - self.Y) 
    
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
  