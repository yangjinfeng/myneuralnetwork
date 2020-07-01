# -*- coding: UTF-8 -*-
'''
Created on 2020年6月30日

@author: yangjinfeng
'''

class HyperParameter(object):
    '''
    classdocs
    '''
    alpha = 0.01 #learning rate
    
    #for L2 regularization
    L2_Reg = True #是否引入L2正则化
    L2_lambd = 0.05 #L2的参数λ
    
    #for Adam
    beta1 = 0.9 # for momentum 
    beta2 = 0.999 # for RMSProp
    epslon = 10e-8 # for RMSProp
    batch_size = 64

    
    