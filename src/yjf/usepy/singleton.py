# -*- coding: UTF-8 -*-
'''
Created on 2020年6月27日

@author: yangjinfeng
'''

class Singleton(object):
    '''
    classdocs
    '''
    
    '''
    创建对象时，先执行__new__函数，在执行__init__函数
    '''
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return None

    def __init__(self,n):
        '''
        Constructor
        '''
        self.name = n
    
    def sss(self):
        print(Singleton._instance)
        print(Singleton._instance.name)
#     @classmethod
#     def factory(cls):
#         return cls.instance
if __name__ == '__main__':
    
    p = Singleton("q")
    q = Singleton("9")
    print(p)
    print(p is q)        
#     q.sss()