# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
from ctypes.test.test_pickling import name

class Student:
    
    def __init__(self,name):
        self.name = name

    def getName(self):
        return self.name
    
    def setName(self):
        self.name = name
    
    S = property(getName,setName)
    
    
if __name__ == '__main__':
    
    p = Student("Tom")
    q = Student("Jack")
    print(p.S)