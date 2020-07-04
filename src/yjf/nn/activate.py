# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np

class Activation(object):
    
    def activate(self,layer):
        pass
    
    def derivative(self,layer):
        pass
    

class Sigmoid(Activation):
    
    def __init__(self):
        self.max = 50
        self.min = -5

    def activate(self,Z):
        
        tempz = np.maximum(self.min,np.minimum(self.max,Z))
#         return 1 / (1 + np.exp(-layer.Z)) 
        return 1 / (1 + np.exp(-tempz)) 
        
    
#     def activate(self,layer):
#         
#         tempz = np.maximum(self.min,np.minimum(self.max,layer.Z))
# #         return 1 / (1 + np.exp(-layer.Z)) 
#         return 1 / (1 + np.exp(-tempz)) 
    
    def derivative(self,layer):
        return layer.A * (1 - layer.A)
    

class Tanh(Activation):
    
#     def activate(self,layer):
#         return np.tanh(layer.Z)
    def activate(self,Z):
        return np.tanh(Z)

    
    def derivative(self,layer):
        return 1 - layer.A * layer.A
        
        
class ReLU(Activation):
    
    
    def activate(self,Z):
#         return (np.abs(Z)+Z) /2
        return np.maximum(0, Z)

#     def activate(self,layer):
# #         return (np.abs(Z)+Z) /2
#         return np.maximum(0, layer.Z)
    
    def derivative(self,layer):
        x = np.copy(layer.Z)   #这是个坑，一定要先拷贝出来
        if layer.plugin is not None:
            x = np.copy(layer.plugin.Z_tilde) 
        x[x>=0] = 1  #
        x[x<0] = 0   #这两行代码的顺序不能变
        return x


class LeakyReLU(Activation):
    
#     def activate(self,layer):
#         return np.maximum(0.01*layer.Z, layer.Z)

    def activate(self,Z):
        return np.maximum(0.01*Z, Z)

    
    def derivative(self,layer):
        x = np.copy(layer.Z)   #这是个坑，一定要先拷贝出来
        if layer.plugin is not None:
            x = np.copy(layer.plugin.Z_tilde) 

        x[x>=0] = 1
        x[x<0] = 0.01
        return x

class Softmax(Activation):
    
#     def activate(self,layer):
#         return np.maximum(0.01*layer.Z, layer.Z)

    def activate(self,Z):
        temp = np.max(Z,axis = 0,keepdims=True)
        a=np.exp(Z-temp)
        s = np.sum(a,axis=0)
        return a/s

    '''
           是个nXn的矩阵,
    i=j时，为 ai(1-ai)
    i<>j时，为 -ai*aj
    '''
    def derivative(self,layer):
        return layer.A * (1 - layer.A)


def factory(name):
    if(name == "sigmoid"):
        return Sigmoid()
    elif(name == "ReLU"):
        return ReLU()
    elif (name == "LeakyReLU"):
        return LeakyReLU()
    elif (name == "tanh"):
        return Tanh()
    elif (name == "softmax"):
        return Softmax()    
    else:
        return Sigmoid()