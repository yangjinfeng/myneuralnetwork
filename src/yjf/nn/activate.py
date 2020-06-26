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
        self.min = -50
        
    
    def activate(self,layer):
        
        tempz = np.maximum(self.min,np.minimum(self.max,layer.Z))
#         return 1 / (1 + np.exp(-layer.Z)) 
        return 1 / (1 + np.exp(-tempz)) 
    
    def derivative(self,layer):
        return layer.A * (1 - layer.A)
    

class Tanh(Activation):
    
    def activate(self,layer):
        return np.tanh(layer.Z)
    
    def derivative(self,layer):
        return 1 - layer.A * layer.A
        
        
class ReLU(Activation):
    
    def activate(self,layer):
#         return (np.abs(Z)+Z) /2
        return np.maximum(0, layer.Z)
    
    def derivative(self,layer):
        x = layer.Z
        x[x<=0] = 0
        x[x>0] = 1
        return x


class LeakyReLU(Activation):
    
    def activate(self,layer):
        return np.maximum(0.01*layer.Z, layer.Z)
    
    def derivative(self,layer):
        x = layer.Z
        x[x<=0] = 0.01
        x[x>0] = 1
        return x


def factory(name):
    if(name == "sigmoid"):
        return Sigmoid()
    elif(name == "ReLU"):
        return ReLU()
    elif (name == "LeakyReLU"):
        return LeakyReLU()
    elif (name == "tanh"):
        return Tanh()
    else:
        return Sigmoid()