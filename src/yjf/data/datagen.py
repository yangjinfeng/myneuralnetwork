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
    def loadCircleDataset(is_plot=False):
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
    
    
    @staticmethod
    def loadClassificationDataset():
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_classification(n_samples=200, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        return train_X, train_Y, test_X, test_Y
        
        
if __name__ == '__main__':      
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadDataset()  
    print(Tr_x.shape)
    print(Tr_y.shape)
    
    print(Tr_x[:,0].reshape(2,1))
    print(Tr_y[:,0].reshape(1,1))
    