# -*- coding: UTF-8 -*-
'''

@author: yangjinfeng
'''
import numpy as np
import time

alpha=0.01
iterations=10000

m=5
#(3,5)
def loadData():
    a=np.array([[1,2,3], [4,5,6],[7,8,9],[100,200,300], [400,500,600]])
    return a.T

def loadDataY():
    return np.array([0,0,0,1,1]).reshape(1,5)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def deva_sigmoid(sigmoid):
    return sigmoid*(1-sigmoid)


#（3,5）
X = loadData()
Y = loadDataY()
#第一层4个节点
W1=np.random.randn(4,3)
B1=np.random.randn(4,5)
#第二层4个节点
W2=np.random.randn(4,4)
B2=np.random.randn(4,5)
#第三层一个节点
W3=np.random.randn(1,4)
B3=np.random.randn(1,5)


tic = time.time()
for it in range(iterations):    
    
    Z1 = np.matmul(W1, X) + B1
    A1 = sigmoid(Z1)
    
    Z2 = np.matmul(W2, A1) + B2
    A2 = sigmoid(Z2)

    Z3 = np.matmul(W3, A2) + B3
    A3 = sigmoid(Z3)
    
    print(A3)

    

    dA3 = -Y/A3 + (1-Y)/(1-A3)
    dZ3 = dA3 * deva_sigmoid(A3)
    dW3 = (1/m) * np.matmul(dZ3, A2.T)
    dB3 = (1/m) * np.sum(dZ3, axis = 1, keepdims=True)
    W3 = W3 - alpha * dW3
    B3 = B3 - alpha * dB3
    
    dA2 = np.matmul(W3.T, dZ3)
    dZ2 = dA2 * deva_sigmoid(A2)
    dW2 = (1/m) * np.matmul(dZ2, A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis = 1, keepdims=True)
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    
    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * deva_sigmoid(A1)
    dW1 = (1/m) * np.matmul(dZ1, X.T)
    dB1 = (1/m) * np.sum(dZ1, axis = 1, keepdims=True)
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1

toc = time.time()

print("time lasted:  " + str(1000*(toc-tic)))