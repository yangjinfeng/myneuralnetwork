# -*- coding: UTF-8 -*-
'''
Created on 2020年6月30日

@author: yangjinfeng
'''

class MyLogger(object):
    '''
    classdocs
    '''


    def __init__(self, filename):
        '''
        Constructor
        '''
        self.file = open("C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\output\\"+filename,"w")
    
    def setNetwork(self,net):
        self.net = net
    
    
    def log(self):
        self.file.write(str(self.net.iterCounter)+","+str(self.net.getLoss())+"\n")
    
    def close(self):
        self.file.close()