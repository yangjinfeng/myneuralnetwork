# -*- coding: UTF-8 -*-
'''
Created on 2020年6月27日

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.wbvector import ParamVectorConverter 

class MockLayer:
    
    def __init__(self):
        self.W = np.random.randn(3,2)
        self.B = np.random.randn(3,1)
        
    

class MockNet:
    def __init__(self):
        self.layers = []
        self.layers.append(MockLayer())
        self.layers.append(MockLayer())
        
    def printnet(self):
        for l in self.layers:
            print(l.W)
            print(l.B)



net=MockNet()
net.printnet()
print(".............")
pvc = ParamVectorConverter(net)
v0 = pvc.toParamVector()
v = np.copy(v0)
v[12] = 888
pvc.fillNet(v)
v2 = pvc.toParamVector()
net.printnet()
# print(v0)
# print(v2)
