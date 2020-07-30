# -*- coding: UTF-8 -*-
'''
Created on 2020年6月22日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.activate import factory
from yjf.nn.cfg import HyperParameter
from yjf.nn.wbvector import ParamVectorConverter
from yjf.nn.plugin import BatchNormPlugin
from yjf.nn.paramupdate import ParameterUpdater

class Layer:
    def __init__(self,N,actvaname,keep_prob):
        self.N = N     #当前层
        self.actva = factory(actvaname)
        self.keep_prob = keep_prob #dropout
        self.net = None
        
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
        self.index = -1
        #for Adam
        self.VdW = None
        self.SdW = None
        self.VdB = None
        self.SdB = None
        self.paramdict = {}
        
        #batch norm plugin
        self.plugin = None
    
    def setLayerIndex(self,index):
        self.index = index

#     def setDataSize(self,dataSize):
#         self.M = dataSize
    
    def setNetWork(self,net):
        self.net = net
        
    def setPreLayer(self,layer):
        self.preLayer = layer
        
    def setNextLayer(self,layer):
        self.nextLayer = layer

    def initialize(self):
        self.W = np.random.randn(self.N,self.preLayer.N) * 0.01;        
        self.VdW = np.zeros((self.N,self.preLayer.N),dtype=float);
        self.SdW = np.zeros((self.N,self.preLayer.N),dtype=float);
#         self.W = np.random.randn(self.N,self.preLayer.N) * np.sqrt(2/(self.preLayer.N));
        self.B = np.zeros((self.N,1),dtype=float);
#         self.B = np.random.randn(self.N,1) * 0.001  #对所有样本，B都是一样的
        self.VdB = np.zeros((self.N,1),dtype=float) 
        self.SdB = np.zeros((self.N,1),dtype=float) 
        
        self.plugin = BatchNormPlugin(self)
        
    def copy(self):
        newlayer = Layer(self.N,"",self.keep_prob)
        newlayer.W = self.W
        newlayer.B = self.B
        newlayer.actva = self.actva
        return newlayer
        
    def forward(self,dropout):#执行测试或者预测的时候，就不做dropout
        
        self.Z = np.matmul(self.W, self.preLayer.A) + self.B  #(n,m)
        self.M = self.Z.shape[1] #样本数
#         self.A = self.activation(self.Z);
        isTraining = dropout
        if self.plugin is not None:
            Z_tilde = self.plugin.plugin_forward(self, isTraining)
            self.A = self.actva.activate(Z_tilde);
        else:
            self.A = self.actva.activate(self.Z);
        #do dropout
        if (dropout):
            probMatrix = np.random.rand(self.A.shape[0],self.A.shape[1]) < self.keep_prob        
            self.A = np.multiply(self.A, probMatrix) / self.keep_prob
    
    
        
    '''
    常规的反向传播，即一次输入全体训练数据
    '''
    def backward(self):
        if(self.isOutputLayer()):
            #尽量不做除法
#             self.dA = self.deriveLoss()
            self.dZ = self.deriveLossByZ()
        else:
#             self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ) #这个bug找了好久
            self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)
            self.dZ =  self.dA * self.actva.derivative(self)        
        
#         self.dZ =  self.dA * self.actva.derivative(self)
        self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
        #L2 regularization
        if(HyperParameter.L2_Reg):
            L2_dW = (HyperParameter.L2_lambd / self.M) * self.W
            self.dW = self.dW + L2_dW
        
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
        
#         weight_decay = 1
#         if(HyperParameter.L2_Reg):
#             weight_decay = 1 - HyperParameter.alpha * HyperParameter.L2_lambd / self.M

        self.W0 = np.copy(self.W) #给前一层计算梯度使用
        self.W = self.W  - HyperParameter.alpha * self.dW
        self.B = self.B  - HyperParameter.alpha * self.dB


    '''
    mini-batch反向传播,参数t为批次，从1开始
    '''
    def backward_mini_batch(self,t):
        if(self.isOutputLayer()):
            #尽量不做除法
#             self.dA = self.deriveLoss()
            self.dZ = self.deriveLossByZ()
        else:
#             self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ) #这个bug找了好久
            self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)
            self.dZ =  self.dA * self.actva.derivative(self)        
        
#         self.dZ =  self.dA * self.actva.derivative(self)
        self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
        #L2 regularization
        if(HyperParameter.L2_Reg):
            L2_dW = (HyperParameter.L2_lambd / self.M) * self.W
            self.dW = self.dW + L2_dW
        
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
        
        #Adam算法
        self.VdW = HyperParameter.beta1 * self.VdW + (1 - HyperParameter.beta1) * self.dW
        self.VdB = HyperParameter.beta1 * self.VdB + (1 - HyperParameter.beta1) * self.dB
        self.SdW = HyperParameter.beta2 * self.SdW + (1 - HyperParameter.beta2) * (self.dW * self.dW)
        self.SdB = HyperParameter.beta2 * self.SdB + (1 - HyperParameter.beta2) * (self.dB * self.dB)
        
        beta1_temp = 1 - np.power(HyperParameter.beta1,t)
        VdW_corrected = self.VdW / beta1_temp
        VdB_corrected = self.VdB / beta1_temp
        beta2_temp = 1 - np.power(HyperParameter.beta2,t)
        SdW_corrected = self.SdW / beta2_temp
        SdB_corrected = self.SdB / beta2_temp
        
        self.W0 = np.copy(self.W) #给前一层计算梯度使用
        self.W = self.W  - HyperParameter.alpha * VdW_corrected / (np.sqrt(SdW_corrected) + HyperParameter.epslon)
        self.B = self.B  - HyperParameter.alpha * VdB_corrected / (np.sqrt(SdB_corrected) + HyperParameter.epslon)

    '''
    mini-batch反向传播,参数t为批次，从1开始
        |<----------------第l层---------------->|<--------------第l+1层----------------->|
    A-->Z-->Z_norm-->Z_tilde-->g(Z_tilde)-->A-->Z-->Z_norm-->Z_tilde-->g(Z_tilde)-->A-->Z
        每一层的反向传播从dA开始
    '''
    def backward_batchnorm(self):
        if(self.isOutputLayer()):
            #尽量不做除法,反向传播的起点最麻烦
#             self.dA = self.deriveLoss()
            dZ_tilde = self.deriveLossByZ()
            self.dZ = self.plugin.plugin_backward(self,dZ_tilde)
        else:
#             self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ) #这个bug找了好久
            self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)  #(n,m)
            dZ_tilde =  self.dA * self.actva.derivative(self)#(n,m)  
            self.dZ = self.plugin.plugin_backward(self,dZ_tilde)
        
#         self.dZ = self.plugin.plugin_backward(self)
        
        self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
        #L2 regularization
        if(HyperParameter.L2_Reg):
            L2_dW = (HyperParameter.L2_lambd / self.M) * self.W
            self.dW = self.dW + L2_dW
        
        self.W0 = np.copy(self.W) #给前一层计算梯度使用
        self.paramdict["W"] = self.W
        #Adam算法
        #B合并到batch norm的beta参数了
        ParameterUpdater.adamUpdate(self.paramdict, "dW", self.dW, "W")
        self.W = self.paramdict["W"]


#     '''
#     mini-batch反向传播,参数t为批次，从1开始
#         |<----------------第l层---------------->|<--------------第l+1层----------------->|
#     A-->Z-->Z_norm-->Z_tilde-->g(Z_tilde)-->A-->Z-->Z_norm-->Z_tilde-->g(Z_tilde)-->A-->Z
#         每一层的反向传播从dA开始
#     '''
#     def backward_batchnorm2(self,t):
#         if(self.isOutputLayer()):
#             #尽量不做除法
#             self.dA = self.deriveLoss()
#         else:
# #             self.dA = np.matmul(self.nextLayer.W.T, self.nextLayer.dZ) #这个bug找了好久
#             self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)  #(n,m)
#         
# #         self.dZ = self.plugin.plugin_backward(self, t)
#         dZ_tilde =  self.dA * self.actva.derivative(self)#(n,m)
#         dgama = np.sum(dZ_tilde * self.plugin.Z_norm, axis = 1,keepdims=True) #(n,1),要不要除以m呢
#         dbeta = np.sum(dZ_tilde, axis = 1,keepdims=True) #(n,1)，要不要除以m呢
#         dZ_norm = dZ_tilde * self.plugin.gamma  #(n,m)
#          
#         '''
#                     计算公式：
#         https://kevinzakka.github.io/2016/09/14/batch_normalization/
#         https://zhuanlan.zhihu.com/p/45614576?utm_source=wechat_session
#         '''
#         m = self.M
#         divied = self.M * np.sqrt(self.plugin.zvar+self.plugin.eps)        
#         self.dZ =  (m * dZ_norm - \
#                np.sum(dZ_norm, axis = 1,keepdims=True) - \
#                self.plugin.Z_norm * np.sum(dZ_norm * self.plugin.Z_norm, axis = 1,keepdims=True)) \
#                / divied
#         
#         self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
#         #L2 regularization
#         if(HyperParameter.L2_Reg):
#             L2_dW = (HyperParameter.L2_lambd / self.M) * self.W
#             self.dW = self.dW + L2_dW
#         
#         self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)
#         self.W0 = np.copy(self.W) #给前一层计算梯度使用
#         self.paramdict["W"] = self.W
#         self.paramdict["B"] = self.B
#         #Adam算法
#         #B合并到batch norm的beta参数了
# #         ParameterUpdater.adamUpdate(self.paramdict, "dW", self.dW, "W", t)
# #         ParameterUpdater.adamUpdate(self.paramdict, "dB", self.dB, "B", t)
#         #Adam算法
#         self.VdW = HyperParameter.beta1 * self.VdW + (1 - HyperParameter.beta1) * self.dW
#         self.VdB = HyperParameter.beta1 * self.VdB + (1 - HyperParameter.beta1) * self.dB
#         self.SdW = HyperParameter.beta2 * self.SdW + (1 - HyperParameter.beta2) * (self.dW * self.dW)
#         self.SdB = HyperParameter.beta2 * self.SdB + (1 - HyperParameter.beta2) * (self.dB * self.dB)
#         
#         beta1_temp = 1 - np.power(HyperParameter.beta1,t)
#         VdW_corrected = self.VdW / beta1_temp
#         VdB_corrected = self.VdB / beta1_temp
#         beta2_temp = 1 - np.power(HyperParameter.beta2,t)
#         SdW_corrected = self.SdW / beta2_temp
#         SdB_corrected = self.SdB / beta2_temp
#         
#         self.W0 = np.copy(self.W) #给前一层计算梯度使用
#         self.W = self.W  - HyperParameter.alpha * VdW_corrected / (np.sqrt(SdW_corrected) + HyperParameter.epslon)
#         self.B = self.B  - HyperParameter.alpha * VdB_corrected / (np.sqrt(SdB_corrected) + HyperParameter.epslon)


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
   
   
'''
Logistics regression output layer
'''   
class BinaryOutputLayer(Layer):
    
    def __init__(self,N,actvaname,keep_prob,cutoff):
        self.cutoff = cutoff
        super(BinaryOutputLayer, self).__init__(N,actvaname,keep_prob)   
    
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
        if(HyperParameter.L2_Reg):
            dataSize = self.Y.shape[1]
            paramVec = ParamVectorConverter.toParamVector(self.net)
            L2loss = (HyperParameter.L2_lambd/(2 * dataSize) ) * (np.sum(paramVec * paramVec)) #L2正则化的损失
            loss = loss + L2loss
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

'''
softmax regression output layer
'''   
class MultiOutputLayer(Layer):
    
    def setCutoff(self,cutoff):
        self.cutoff = cutoff
    
    '''
        最后一层计算样本的损失,包括正则项的损失
    '''
    def loss(self):
        eachloss  = np.multiply(-np.log(self.A), self.Y)
        loss = (1/eachloss.shape[1]) * np.sum(eachloss, axis = 1, keepdims=True)[0][0]
        if(HyperParameter.L2_Reg):
            dataSize = self.Y.shape[1]
            paramVec = ParamVectorConverter.toParamVector(self.net)
            L2loss = (HyperParameter.L2_lambd/(2 * dataSize) ) * (np.sum(paramVec * paramVec)) #L2正则化的损失
            loss = loss + L2loss
        return loss
    
    '''
    下面的计算不对，求导的结果是一个矩阵.
    '''
    def deriveLoss(self):
#         self.dA = -self.Y/self.A
        eps = 1e-8 #防止分母为零
        self.dA = -self.Y/(self.A+eps)
        return self.dA

    '''
    dLoss/dA * dA/dZ
    a-y
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
        zmax = np.max(prdct,axis = 0,keepdims=True)
        tmp = prdct-zmax
        tmp[tmp>=0]=1
        tmp[tmp<0]=0
        return tmp
