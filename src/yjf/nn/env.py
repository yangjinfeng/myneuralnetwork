# -*- coding: UTF-8 -*-
'''
Created on 2020年7月4日

@author: yangjinfeng
'''

class GlobalStatus(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.currentEpoch = -1      #迭代次数，epoch
        self.currentBatch = -1      #当前批次，t
        
globalEnv = GlobalStatus()        