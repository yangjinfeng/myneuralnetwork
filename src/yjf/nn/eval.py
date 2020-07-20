# -*- coding: UTF-8 -*-
'''
Created on 2020年6月29日

@author: yangjinfeng
'''
import numpy as np

class EvalMatrix:
    def __init__(self,TP,TN,FP,FN):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
    
    def accuracy(self):
        right = self.TN + self.TP
        return right * 1.0 / (self.TN + self.TP + self.FP + self.FN)

    # def accuracy_p(self):
    #     return self.TP * 1.0 / (self.TP + self.FN)
        
    # def accuracy_n(self):
    #     return self.TN * 1.0 / (self.TN + self.FP)
    
    def toMatrix(self):
        return np.array([[self.TP ,self.FP],[self.FN ,self.TN]])

    def precision(self):
        return self.TP * 1.0 / (self.TP + self.FP)

    def recall(self):
        return self.TP * 1.0 / (self.TP + self.FN)

    def F_measure(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r)

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
        TP = 0  # True Positive：本来是正样例，分类成正样例。
        TN = 0  # True Negative：本来是负样例，分类成负样例。
        FP = 0  # False Positive ：本来是负样例，分类成正样例，通常叫误报。
        FN = 0  # False Negative：本来是正样例，分类成负样例，通常叫漏报。
        
#         print(self.golden.shape)
#         print(self.prediction.shape)
        for i in range(self.golden.shape[1]):
            p = self.prediction[0][i]
            g = self.golden[0][i]
            if p == g:
                if g == 0:
                    TN = TN +1
                else:
                    TP = TP + 1
            else:
                if g == 0:
                    FP = FP +1
                else:
                    FN = FN + 1
                    
        return EvalMatrix(TP,TN,FP,FN)
    
    
    
    
    
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