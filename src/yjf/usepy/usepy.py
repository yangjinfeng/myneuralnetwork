# -*- coding: UTF-8 -*-
'''
Created on 2020年6月23日

@author: yangjinfeng
'''
import numpy as np
from builtins import getattr
import types
from yjf.data.datagen import DataGenerator
import sklearn
import sklearn.datasets
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import io
import sys


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


def testArraySplit():
#     a=np.random.randint(1,9,(4,4))
    a = np.array([[2,6,7,3]\
                 ,[2,4,3,5]\
                 ,[3,2,5,1]\
                 ,[1,4,4,3]])
    x = np.array([0,2,1])
    print(a[x])  #取x给出的位置对应的行，组成新的数组，即取出第0行、第2行、第1行
    '''
        输出如下：
        [[2 6 7 3]
         [3 2 5 1]
         [2 4 3 5]]
    '''    
    x = np.array([[0,2,1],[2,3,0]])
    print(a[x])  #取x给出的位置对应的行，组成新的数组，即取出第0行、第2行、第1行,组成一个新的数组，取出第2行、第3行、第0行,组成另一个新的数组，最后组成三维数组
    '''
        输出如下：
        [[[2 6 7 3]
          [3 2 5 1]
          [2 4 3 5]]
        
         [[3 2 5 1]
          [1 4 4 3]
          [2 6 7 3]]]
    '''    
    print(a[[0,2,1],[2,3,0]]) #依次取出（0,2）、（2,3）、（1,0）的值组成一个向量，输出[7 1 2]
    print(a[0:2,0:3])#行的位置是0至2，列的位置是0至3，不包括最后一个位置，截取出一个小矩阵
    
    print(a[[0,2,1,3]][[2,3,0]]) #先取一次a[[0,2,1,3]]，效果类似于a[x]，再用同样的规则用[2,3,0]取一次
    
    print(a[2])
    print(a[[2]])
    print(a[2][0])
    print(a[[2]][0])
    print(a[[2]][[0]])


def testreshape():
    a = np.random.randn(20)
    b =a.reshape(-1,1)  #等价于 a.reshape(len(a),1)
    print(b.shape)
    # print(b)

    x=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(x[0:2,-1])   #前两行的最后一列
    
def testSqueeze():
    e= np.arange(10)
    a = e.reshape(1,1,10)
    print(a.shape)
    b = np.squeeze(a)
    print(b.shape)

'''
二维数组的堆叠
'''
def testvstack():
    a1=np.array([[1,1,1],[1,1,1]])
    a2=np.array([[2,2,2],[2,2,2]])
    v = np.vstack((a1,a2)) #纵向堆叠
    v2 = np.concatenate((a1,a2),axis=0)
    h = np.hstack((a1,a2)) #横向堆叠
    h2 = np.concatenate((a1,a2),axis=1)
    print(v)
    print(v2)
    print(h)
    print(h2)

    '''
    由二维数组构造三位矩阵，在z方向堆叠
    '''    
def teststack():
    #第一种方法，使用dstack方法
    f1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ff = np.dstack((f1,f1))
    fff = np.dstack((ff,f1))
    # ff = np.stack((f1,f1),axis=2)
    print(fff)
    print(fff[:,:,0])   
    print("---------------")
    #第二种方法，先构造出空矩阵，然后再z方向切片，对切片赋值
    f1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    fa = np.zeros((3,3,3),dtype=int)
    fa[:,:,0]=f1
    fa[:,:,1]=f1
    fa[:,:,2]=f1
    print(fa)
    print(fa[:,:,0])   


def testOnehotencoder():
    oh = OneHotEncoder()
    a=np.array(["A","A","B","C","A","B","C","C","D"])
    ohc=oh.fit_transform(a.reshape(-1, 1))
    print(ohc.A)
    print(oh.categories_)
    
def testDummy():
    oh = OneHotEncoder()
    db = pd.read_csv("file:\\C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\data\\extracted_variable_c12.csv")
    c1=db["concourse_c1"]
    #c1.describe()
    c1array = c1.to_numpy()
    ohc=oh.fit_transform(c1array.reshape(-1, 1))
#     print(ohc.A)
    dummy=pd.get_dummies(db,"concourse_c1")[0:3] #这种方式内存开销太大
    print(dummy.shape)
    
def testtransform():
    db = pd.read_csv("file:\\C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\data\\extracted_variable_c12.csv")
    data = db["has_c2"].to_numpy()
    testdata = data.reshape(-1,1)
    le= LabelEncoder()
    lec = le.fit_transform(testdata)
    print(testdata)
    print(lec)


def softmax(Z):
    temp = np.max(Z,axis = 0,keepdims=True)
    a=np.exp(Z-temp)
    s = np.sum(a,axis=0)
    return a/s


def testRandomChoice():
    # np.random.seed(1)
    # x=np.random.randn(10)
    # print(x)
    # c = np.random.choice(x,2)
    # print(c)
    dic = ["哈","尔","滨","真","凉","快"]
    pv = np.array([7,5,10.2,3,9,6])
    p = softmax(pv)
    for i in range(10):
        c = np.random.choice(dic,3,p=p)
        print(c)

def testArrayConcate():
    tp = (1,2,3,4)
    x = tp[:-1] + (0,)  #把前三个取出来，再拼接元组(0,)，此处逗号不能少
    print(x)
    a = [1,2,3]
    b = a + [4]
    print(b)


def testT():
    x = np.array([[1,2,3],[4,5,6]])
    print(x)
    x2 = x.T
    print(x2)
    x3 = x2.T
    print(x3)

def testConv():
    x = np.array([[1,2,3],[4,5,6]])
    np.convolve()   
    
def funparam(x,*args,**kwargs):
    print(type(args)) #tuple
    print(type(kwargs)) #dict
    
if __name__ == '__main__':
#     testObjattr()
#     testmultipy()
#     testArrayparam()
#     testDict()
#     testglobals()
#     testClassificationData()
#     g = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,0]]).T    
#     p = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0],[1,0,0]]).T
#     r = evalN(g,p)
#     print(r)
    # testArraySplit()
    # testreshape()
#     testvstack()
#     testOnehotencoder()
    # testtransform()
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#     testRandomChoice()
#     testT()
#     testArrayConcate()
#     testSqueeze()
#     teststack()
    funparam(1)