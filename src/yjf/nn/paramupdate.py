# -*- coding: UTF-8 -*-
'''
Created on 2020��7��3��

@author: yangjinfeng
'''
import numpy as np
from yjf.nn.cfg import HyperParameter
class ParameterUpdater(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    @staticmethod
    def adamUpdate(paramdict,dparam_name, dparam, param_name,t):
        Vd = "V"+dparam_name
        Sd = "S"+dparam_name
        param = paramdict[param_name]
        if Vd not in paramdict:
            paramdict[Vd] = np.zeros((param.shape[0],param.shape[1]),dtype=float)
        if Sd not in paramdict:
            paramdict[Sd] = np.zeros((param.shape[0],param.shape[1]),dtype=float)
        
        paramdict[Vd] = HyperParameter.beta1 * paramdict[Vd] + (1 - HyperParameter.beta1) * dparam
        paramdict[Sd] = HyperParameter.beta2 * paramdict[Sd] + (1 - HyperParameter.beta2) * (dparam * dparam)
        
        beta1_temp = 1 - np.power(HyperParameter.beta1,t)
        Vd_corrected = paramdict[Vd] / beta1_temp
        beta2_temp = 1 - np.power(HyperParameter.beta2,t)    
        Sd_corrected = paramdict[Sd] / beta2_temp
        
        paramdict[param_name] = paramdict[param_name]  - HyperParameter.alpha * Vd_corrected / (np.sqrt(Sd_corrected) + HyperParameter.epslon)

           