# -*- coding: UTF-8 -*-
'''
Created on 2020年6月29日

@author: yangjinfeng
'''
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

class DataGenerator(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    @staticmethod
    def loadDataset(is_plot=False):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
        # Visualize the data
        if is_plot:
            plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40);
            plt.show()
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        return train_X, train_Y, test_X, test_Y

if __name__ == '__main__':      
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadDataset()  
    print(Tr_x.shape)
    print(Tr_y.shape)
    
    print(Tr_x[:,0].reshape(2,1))
    print(Tr_y[:,0].reshape(1,1))
    