# -*- coding: UTF-8 -*-
'''
Created on 2020年6月29日

@author: yangjinfeng
'''

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