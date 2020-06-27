# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
from ctypes.test.test_pickling import name

class Student:
    
    def __init__(self,name):
        self.qwe=None
        self.name = name

    def getName(self):
        return self.name
    
    def settName(self,name):
        self.name = name
    
    S = property(getName,settName)
    
    
if __name__ == '__main__':
    
    p = Student("Tom")
    q = Student("Jack")
    p.S="123"
    print(p.S)
    if(not p.qwe):
        print("p.qwe is none")
    print(p.qwe is None)
    if(not hasattr(p, "qwe2")):
        print("p hasn't qwe2")
    s = object.__new__(Student)
    s.__init__('s')
    print(s.getName())