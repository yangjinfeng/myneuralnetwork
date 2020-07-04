# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from builtins import getattr
import types
from yjf.data.datagen import DataGenerator

class Student:
    
    def __init__(self,name):
        self.qwe=np.array([1,2])
        self.name = name

    def getName(self):
        return self.name
    
    def settName(self,name):
        self.name = name
    
    S = property(getName,settName)


def helloworld( s):
    print("hello " +s)

'''
可以动态地给一个对象绑定成员变量或者一个函数
'''    
def testObjattr():
    s = Student("a")
    s.age=10
    s.f = helloworld
    print(s)
    print(s.age)
    s.f("deep learning")
    for p in s.__dict__:
        print(p)
    getattr(s, "f")("111") #hello 111
    print(type(getattr(s, "f")))  # 输出  <class 'function'>
    print(type(getattr(s, "settName")))  # 输出  <class 'method'>
    print(type(getattr(s, "name")))  # 输出  <class 'method'>
    print(type(getattr(s, "age")))  # 输出  <class 'method'>
    
    if type(s.f) == types.FunctionType:
        s.f("qwe")
    for a in dir(s):
        print(a)
def testmultipy():
    a0 = np.array([[1,2,3],[4,5,6]])
    a1 = np.array([[1,2,3],[4,5,6]])
    a2 = np.array([[1,2],[4,5],[7,8]])
#     a3=np.dot(a1,a2)
#     a3=np.matmul(a1,a2)
#     a3=a1 @a2
#     a3=np.dot(a1,a2)    
#     print(a0 * a1)
    print(a0)
    print(np.sum(a0, axis = 1,keepdims=True))
    print(np.sum(a0, axis = 0,keepdims=True))
    
def testNew():
    s = object.__new__(Student)
    s.__init__('s')
    print(s.getName())


    
def  funcparam(s=0,w=True):
    print(s)
    print(w)    
    
def testMeanVariance():
    v=np.random.randn(20)
    m=np.mean(v,axis=0,keepdims=True)
    var = np.sum((v-m)*(v-m))/len(v)
    vara = np.var(v,keepdims=True)
    print(var)
    print(vara)

def testDict():
    a = np.array([[1,2],[3,4]])
    a2 = a + 2 #得到一个新的数组
    print(a is a2) #False
    
    dict = {"q":a}
    b=dict["q"]
    print(a is b)
    
    b[0][0] = b[0][0] + 10
    print(a is b)  #True
    
    b = b + 10
    print(a)
    print(a is b)
    print(a is dict["q"]) #dict["q"]没有覆盖，指向的引用还是a
     
    
    dict["q"] = dict["q"]+10 #覆盖了原来的数组
    print(dict["q"] is b)
    print(b)
    
'''
这个要引起注意，数组是拷贝传进去的，不是传递引用
'''    
def funcarray2(array):
    array = array+ 10
   
def testArrayparam():     
    a= np.array([1,2,3,4])
    funcarray2(a)
    print(a)
    
def testglobals():
    print(globals())
    
def testClassificationData():
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadNClassificationDataset(4)
    print(T_x.shape)
    print(T_y.shape)
    print(T_y)
    
def evalN(golden,prediction):
    s = 0
    for i in range(golden.shape[1]):
        s = s + np.dot(golden[:,i],prediction[:,i])
    return s * 1.0 / golden.shape[1]    
    
if __name__ == '__main__':
#     testObjattr()
#     testmultipy()
#     testArrayparam()
#     testDict()
#     testglobals()
#     testClassificationData()
    g = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,0]]).T    
    p = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,0,0]]).T
    r = evalN(g,p)
    print(r)