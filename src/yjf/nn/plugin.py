# -*- coding: UTF-8 -*-
'''
Created on 2020年7月3日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.cfg import HyperParameter
from yjf.nn.paramupdate import ParameterUpdater

class BatchNormPlugin(object):
    '''
    classdocs
    '''


    def __init__(self, layer):
        '''
        Constructor
        '''
        self.layer = layer
        '''
        gamma初始化1，beta初始化0相当于不做缩放，下面这个链接的代码可以印证
        https://www.jianshu.com/p/fa06399863ad
        '''
        self.beta =  np.zeros((self.layer.N,1),dtype=float);
        self.gamma =  np.ones((self.layer.N,1),dtype=float);
        
        self.zmean = np.zeros((self.layer.N,1),dtype=float);
        self.zvar = np.zeros((self.layer.N,1),dtype=float);
        self.Z_norm = None
        self.Z_tilde = None
        self.eps = 1e-8
        self.momentum = 0.9
        
        self.paramdict={"beta":self.beta,"gamma":self.gamma}
        
        
        '''
        isTraining=True，训练； False，测试或者预测
        '''
    def plugin_forward(self,layer,isTraining):
        Z = layer.Z
        if isTraining :
            sample_mean = np.mean(Z, axis=1, keepdims=True)
            sample_var = np.var(Z, axis=1, keepdims=True)
            self.Z_norm = (Z - sample_mean) / np.sqrt(sample_var + self.eps)
            self.Z_tilde = self.gamma * self.Z_norm + self.beta # Z_tilde也就是Z_~
        
            # update moving average
            self.zmean = self.momentum * self.zmean + (1-self.momentum) * sample_mean
            self.zvar = self.momentum * self.zvar + (1-self.momentum) * sample_var
        else:
            self.Z_norm = (Z - self.zmean) / np.sqrt(self.zvar + self.eps)
            self.Z_tilde = self.gamma * self.Z_norm + self.beta # Z_tilde也就是Z_~

        return self.Z_tilde
    
    def plugin_backward(self,layer,t):
        dZ_tilde =  layer.dA * layer.actva.derivative(layer)#(n,m)
        dgama = np.sum(dZ_tilde * self.Z_norm, axis = 1,keepdims=True) #(n,1),要不要除以m呢
        dbeta = np.sum(dZ_tilde, axis = 1,keepdims=True) #(n,1)，要不要除以m呢
        dZ_norm = dZ_tilde * self.gamma  #(n,m)
        
        '''
                    计算公式：
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        https://zhuanlan.zhihu.com/p/45614576?utm_source=wechat_session
        '''
        m = layer.dA.shape[1]
        divied = m * np.sqrt(self.zvar+self.eps)        
        layer.dZ =  (m * dZ_norm - \
               np.sum(dZ_norm, axis = 1,keepdims=True) - \
               self.Z_norm * np.sum(dZ_norm * self.Z_norm, axis = 1,keepdims=True)) \
               / divied
        #Adam算法
        ParameterUpdater.adamUpdate(self.paramdict, "dgama", dgama, "gamma", t)
        ParameterUpdater.adamUpdate(self.paramdict, "dbeta", dbeta, "beta", t)
        self.gamma = self.paramdict["gamma"]
        self.beta = self.paramdict["beta"]
        return layer.dZ
        
        