# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''
import numpy as np
import yjf.cnn.CnnUtil as util
from yjf.nn.activate import factory
from yjf.nn.wbvector import ParamVectorConverter
from yjf.nn.cfg import HyperParameter

class Layer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.type = "conv"
        self.plugin = None
        self.net = None
        
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def setNetWork(self,net):
        self.net = net


    def setPreLayer(self,layer):
        self.preLayer = layer
        
    def setNextLayer(self,layer):
        self.nextLayer = layer
    
    def isOutputLayer(self):
        return False

    def isInputLayer(self):
        return False
    
class ConvLayer(Layer):   
    #ConvLayer({"size":3,"stride":1,"padding":"SAME","count":3},"ReLU")  
    #size=3,stride=1,padding="valid",count=3
    #channels取决于上一层
    def __init__(self, filter_dict,actvaname):
        self.filter_dict = filter_dict
        self.actva = factory(actvaname)
        self.preLayer = None
        self.nextLayer = None
        self.A = None
        self.Z = None
        self.A1_shape = None #单个图片的
        super(ConvLayer, self).__init__()

    def initialize(self):
        channels = self.preLayer.A1_shape[2] 
        size = self.filter_dict["size"] #卷积核的大小，为一个正方形
        filter_count = self.filter_dict["count"]#卷积后的通道数
        #就是过滤器filter的矩阵，实质就是权重参数
        self.W = np.random.randn(size,size,channels,filter_count)*0.01    
        #确定B的shape，这其实也是本层输出的shape    
        stride = self.filter_dict["stride"]
        padding = self.filter_dict["padding"]
        h,w = util.getPadedOutShape(self.preLayer.A1_shape,size,stride,padding)
        self.A1_shape=(h,w,filter_count)
        self.A1_size = h * w * filter_count
#         self.B = np.random.randn(h,w,filter_count) #初始化B
        self.B = np.zeros((1,1,filter_count),dtype=float) #初始化B

        

    def forward(self):
        stride = self.filter_dict["stride"]
        padding = self.filter_dict["padding"]
        preA = self.preLayer.A
        self.Z = util.conv_forward_tf(preA, self.W, stride, padding) + self.B
        self.A = self.actva.activate(self.Z)
    
    '''
    dA是每一层反向传播的输入，每一层反向传播最后计算上一层的dA
    '''
    def backward(self):
        
        self.dZ =  self.dA * self.actva.derivative(self)        
        
        self.preLayer.dA = util.conv_backprop_input_tf(self.preLayer.A.shape, 
                                              self.W, 
                                              self.dZ, 
                                              self.filter_dict["stride"], 
                                              self.filter_dict["padding"])

        self.dW = util.conv_backprop_filter_tf(self.preLayer.A, 
                                               self.W.shape, 
                                               self.dZ, 
                                               self.filter_dict["stride"], 
                                               self.filter_dict["padding"])
        
        batch = self.dZ.shape[0]
        self.dB = (np.sum(self.dZ,axis = (0,1,2)).reshape(self.B.shape))/batch
        
        #L2 regularization
        if(HyperParameter.L2_Reg):
            L2_dW = (HyperParameter.L2_lambd / batch) * self.W
            self.dW = self.dW + L2_dW

        self.W = self.W  - HyperParameter.alpha * self.dW
        self.B = self.B  - HyperParameter.alpha * self.dB


class FCCLayer(Layer):    
    #FCCLayer(100,"ReLU") 
    def __init__(self,N,actvaname):
        self.actva = factory(actvaname)
        self.N = N
        self.A1_size = N
        super(FCCLayer, self).__init__()        

    def initialize(self):        
        self.W = np.random.randn(self.N,self.preLayer.A1_size) * 0.01;
        self.B = np.zeros((self.N,1),dtype=float);
        
    def forward(self):
        preA = self.preLayer.A
#         preA1_size = self.preLayer.A1_size
        if isinstance(self.preLayer, ConvLayer) or isinstance(self.preLayer, PoolLayer):
            flattened_preA = util.matrix_to_flattened(preA) #上一层是矩阵，要用扁平化的数组
            self.preLayer.flattened_A = flattened_preA
                
            self.Z = np.matmul(self.W, flattened_preA) + self.B  #(n,m)    
            self.A = self.actva.activate(self.Z)
            
        elif isinstance(self.preLayer, FCCLayer):    
            self.Z = np.matmul(self.W, preA) + self.B  #(n,m)    
            self.A = self.actva.activate(self.Z)
#         preA = self.preLayer.A.flatten()
#         self.Z = np.matmul(self.W, preA.reshape((len(preA),1))) + self.B  #(n,m)
#         self.A = self.actva.activate(self.Z)
    
    def backward(self):
        if(self.isOutputLayer()):
            self.dZ = self.deriveLossByZ()
        else:
#             self.dA = np.matmul(self.nextLayer.W0.T, self.nextLayer.dZ)
            self.dZ =  self.dA * self.actva.derivative(self)        
        
        self.M = self.Z.shape[1] #样本数
        pre_dA = np.matmul(self.W.T, self.dZ)#应该是这样的，先放这里，后续再改
        if isinstance(self.preLayer, ConvLayer) or isinstance(self.preLayer, PoolLayer):
#             self.preLayer.dA = pre_dA.reshape(self.preLayer.A.shape)
            self.preLayer.dA = util.flattened_to_matrix(pre_dA, self.preLayer.A.shape)
            self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.flattened_A.T)#上一层是矩阵，要用扁平化的数组
        else:                
            self.preLayer.dA = pre_dA
            self.dW = (1/self.M) * np.matmul(self.dZ, self.preLayer.A.T)
        
        
        
        #L2 regularization
        if(HyperParameter.L2_Reg):
            L2_dW = (HyperParameter.L2_lambd / self.M) * self.W
            self.dW = self.dW + L2_dW
        
        self.dB = (1/self.M) * np.sum(self.dZ, axis = 1, keepdims=True)

        self.W = self.W  - HyperParameter.alpha * self.dW
        self.B = self.B  - HyperParameter.alpha * self.dB

  
        
class PoolLayer(Layer):
    
    #PoolLayer({"method":"MAX","size":3,"stride":1,"padding":"SAME"})
    def __init__(self, pool_dict):
        self.pool_dict = pool_dict

    
    def initialize(self):
        self.pool_method = self.pool_dict["method"]
        self.pool_size = self.pool_dict["size"] #卷积核的大小，为一个正方形
        self.stride = self.pool_dict["stride"]
        self.padding = self.pool_dict["padding"]
        h,w = util.getPadedOutShape(self.preLayer.A1_shape,self.pool_size,self.stride,self.padding)
        self.A1_shape=(h,w,self.preLayer.A1_shape[2])
        self.A1_size = h * w * self.preLayer.A1_shape[2]

    def forward(self):
        if self.pool_method == "MAX":
            max_out,max_pos = util.pool_forward_max_argmax_tf_v(self.preLayer.A, self.pool_size, self.stride, self.padding)
            self.A = max_out
            self.pos_tuple = max_pos
        else:
            self.A = util.pool_forward_tf(self.preLayer.A, self.pool_size, self.stride, self.padding, "AVG")
            
    
    
    def backward(self):
        if self.pool_method == "MAX":
            self.preLayer.dA = util.max_pooling_backprop(self.preLayer.A.shape, self.dA, self.pos_tuple)
        else:
            self.preLayer.dA = util.avg_pooling_backprop(self.preLayer.A.shape, self.pool_size, self.dA, self.stride, self.padding)
        
    
class InputLayer(Layer):
    
    def __init__(self):
        pass
    
    def setInputData(self,data):
        self.A = data*1.0 #tensorflow在计算反向传播的时候要求不能是整数（uint8）
        self.A1_shape = data[0].shape

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
#         print("predict:\n")
#         print(prdct)
#         print("z:\n")
#         print(self.Z)
        prdct[prdct>self.cutoff]=1
        prdct[prdct<=self.cutoff]=0
        return self.A,prdct
