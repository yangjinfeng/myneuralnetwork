# -*- coding: UTF-8 -*-
'''
Created on 2020年6月27日

@author: yangjinfeng
'''
import numpy as np

class ParamVectorConverter(object):
    '''
    classdocs
    '''


#     def __init__(self, net):
#         self.net = net
    
    @staticmethod    
    def toParamVector(net):
        vec = np.array([])
        for layer in net.layers:
            if hasattr(layer,'W'):
                vec = np.concatenate((vec,layer.W.flatten()))
            if hasattr(layer,'B'):
                vec = np.concatenate((vec,layer.B.flatten()))
        return vec
    
    @staticmethod           
    def toGradVector(net):
        vec = np.array([])
        for layer in net.layers:
            vec = np.concatenate((vec, layer.dW.flatten()))
            vec = np.concatenate((vec, layer.dB.flatten()))
        return vec
    
    
    @staticmethod           
    def fillNet(net,paramVec):
        begin = 0
        length = 0
        for layer in net.layers:
            length = layer.W.shape[0]*layer.W.shape[1]
            layer.W = paramVec[begin:begin+length].reshape(layer.W.shape)
            begin = begin+length 
            length = layer.B.shape[0]*layer.B.shape[1]
            layer.B = paramVec[begin:begin+length].reshape(layer.B.shape)
            begin = begin+length 
    
    @staticmethod
    def computeLoss(net,paramVec):
            ParamVectorConverter.fillNet(net, paramVec)
            net.forward()
            loss = net.getLoss()
            return loss


