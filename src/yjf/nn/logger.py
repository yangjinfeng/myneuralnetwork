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
        iter = str(self.net.iterCounter)
        self.file.write(iter+","+str(self.net.getLoss())+"\n")
#         self.file.write(iter+","+str(self.net.getPrediction())+"\n\n")
        
    
    def close(self):
        self.file.close()