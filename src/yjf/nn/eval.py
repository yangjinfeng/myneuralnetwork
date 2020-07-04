# -*- coding: UTF-8 -*-
'''
Created on 2020年6月29日

@author: yangjinfeng
'''
import numpy as np

class Evaluator(object):
    '''
    classdocs
    '''


    def __init__(self, golden,prediction):
        '''
        Constructor
        '''
        self.golden = golden.astype(int)
        self.prediction = prediction.astype(int)
        
        
    def eval(self):
        right = 0
        print(self.golden)
        print(self.prediction)
        for i in range(self.golden.shape[1]):
            if self.golden[0][i] == self.prediction[0][i]:
                right = right+ 1
        return right * 1.0 / self.golden.shape[1]
    
    def evalN(self):
        print(self.golden.shape)
        print(self.prediction.shape)
        s = 0
        for i in range(self.golden.shape[1]):
            c1 = self.golden[:,i]
            c2= self.prediction[:,i]
            ji = np.dot(c1,c2)
            s = s + ji
        return s * 1.0 / self.golden.shape[1]