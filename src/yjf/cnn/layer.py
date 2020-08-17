# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''
import numpy as np
import yjf.cnn.CnnUtil as util
from yjf.nn.activate import factory


class Layer(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.type = "conv"
        
    
    def forward(self):
        pass

    def setPreLayer(self,layer):
        self.preLayer = layer
        
    def setNextLayer(self,layer):
        self.nextLayer = layer

class ConvLayer(Layer):     
    #size=3,stride=1,padding="valid",count=3
    #channels取决于上一层
    def __init__(self, filter_dict,actvaname):
        self.filter_dict = filter_dict
        self.actva = factory(actvaname)
        self.preLayer = None
        self.nextLayer = None
        self.A = None
        self.Z = None
        self.A_shape = None

    def initialize(self):
        (pre_h,pre_w,channels) = self.preLayer.A_shape #单个图片的
        self.W = [] #就是过滤器filter的矩阵，实质就是权重参数
        size = self.filter_dict["size"]
        filter_count = self.filter_dict["count"]
#         for i in range(filter_count):
#             self.W.append(np.random.randn(size,size,channels)) #初始化W
        self.W = np.random.randn(size,size,channels,filter_count)    
        #确定B的shape，这其实也是本层输出的shape    
        stride = self.filter_dict["stride"]
        padding = self.filter_dict["padding"]
        pad = 0
        if padding == "SAME":
            pad=int((size-1)/2)            
        h = int((pre_h+2*pad - size)/stride + 1)
        w = int((pre_w+2*pad - size)/stride + 1)
        self.A_shape=(h,w,filter_count)
        self.A_size = h * w * filter_count
        self.B = np.random.randn(h,w,filter_count) #初始化B


        

    def forward(self):
        stride = self.filter_dict["stride"]
        padding = self.filter_dict["padding"]
        preA = self.preLayer.A
#         m = len(preA)
#         z_data = np.empty((m,) + self.A_shape, dtype = float) 
#         for i in range(m):
#             z_data[i] = util.conv_forward2(preA[i], self.W, stride, padding)+self.B
#         self.Z = z_data
        self.Z = util.conv_forward_tf(preA, self.W, stride, padding) + self.B
        self.A = self.actva.activate(self.Z)
        

class FCCLayer(Layer):     
    def __init__(self,N,actvaname):
        self.actva = factory(actvaname)
        self.N = N
        self.A_size = N        

    def initialize(self):        
        self.W = np.random.randn(self.N,self.preLayer.A_size) * 0.01;
        self.B = np.random.randn(self.N,1) * 0.001 
        
    def forward(self):
        preA = self.preLayer.A
        preA_size = self.preLayer.A_size
        if isinstance(self.preLayer, ConvLayer):
            m = len(preA)
            flattened_preA = np.empty((preA_size,m), dtype = float)
            for i in range(m):
                flattened_preA[:,i] = preA[i].flatten()
                
            self.Z = np.matmul(self.W, flattened_preA) + self.B  #(n,m)    
            self.A = self.actva.activate(self.Z)
            
        elif isinstance(self.preLayer, FCCLayer):    
            self.Z = np.matmul(self.W, preA) + self.B  #(n,m)    
            self.A = self.actva.activate(self.Z)

#         preA = self.preLayer.A.flatten()
#         self.Z = np.matmul(self.W, preA.reshape((len(preA),1))) + self.B  #(n,m)
#         self.A = self.actva.activate(self.Z)
        
class PoolLayer(Layer):
    def __init__(self, params):
        pass

    
class InputLayer(Layer):
    
    def __init__(self):
        pass
    
    def setInputData(self,data):
        self.A = data
        self.A_shape = data[0].shape

    def isInputLayer(self):
        return True
    

'''
Logistics regression output layer
'''   
class BinaryOutputLayer(FCCLayer):
    
    def __init__(self,N,actvaname,cutoff):
        self.cutoff = cutoff
        super(BinaryOutputLayer, self).__init__(N,actvaname)   
    
#     def __init__(self):
#         self.cutoff = 0.5
#         super(Layer, self).__init__(1,"sigmoid",1)
        
    def setCutoff(self,cutoff):
        self.cutoff = cutoff
    
    '''
        最后一层计算样本的损失,包括正则项的损失
    '''
    def loss(self):
        eachloss  = np.multiply(-np.log(self.A), self.Y) + np.multiply(-np.log(1 - self.A), 1 - self.Y)
        loss = (1/eachloss.shape[1]) * np.sum(eachloss, axis = 1, keepdims=True)[0][0]
        return loss
    
    '''
    a=y^=σ(z)
    Logistic regression: loss(a,y) = -( ylog(a) + (1-y)log(1-a)) 
    da = derivative(loss(a,y)) = -y/a + (1-y)/(1-a)
          输出层的损失函数对A求导，计算出dA即可，启动反向传播
    '''
    def deriveLoss(self):
#         self.dA = -self.Y/self.A + (1-self.Y)/(1-self.A)
        eps = 1e-8 #防止分母为零
        self.dA = -self.Y/(self.A+eps) + (1-self.Y)/(1-self.A+eps)
        return self.dA

    '''
    dLoss/dA * dA/dZ
    (-y/a + (1-y)/(1-a)) * (a-a^2) = a-y
    '''
    def deriveLossByZ(self):
        return self.A - self.Y

    
    def setExpectedOutput(self,output):
#         self.Y = np.array([0,0,0,1,1]).reshape(1,5)
        self.Y = output

    def isOutputLayer(self):
        return True
    
    def predict(self):
        prdct = np.copy(self.A)
        prdct[prdct>self.cutoff]=1
        prdct[prdct<=self.cutoff]=0
        return prdct
