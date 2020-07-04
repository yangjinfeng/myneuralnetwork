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

    '''
        np.eye(n_labels)[target_vector]
        This is indexing a NumPy array using another array as described 
        here: https://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays
        The first array is the Identity matrix of size n_labels. 
        The second array selects the one-hot row corresponding to each target. 
    '''
    @staticmethod
    def labelToOnehot(Y):
        num_class = np.max(Y) + 1
        #b[range(len(x)), x] = 1
        b = np.eye(num_class)[Y]
#         return b
        onehot = b[:][:][0].T
        return onehot
#         print(Y.shape)
#         print(onehot)
        

    @staticmethod
    def loadNClassificationDataset(classes_num,train_num,test_num):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_classification(n_samples=train_num, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=classes_num, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        train_Y = DataGenerator.labelToOnehot(train_Y)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_classification(n_samples=test_num, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=classes_num, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        test_Y = DataGenerator.labelToOnehot(test_Y)
        
        return train_X, train_Y, test_X, test_Y
        
        
if __name__ == '__main__':      
    Tr_x, Tr_y, T_x,T_y= DataGenerator.loadNClassificationDataset(4,10,5)  
    
    print(Tr_x.shape)
    print(Tr_y.shape)
    print(T_x.shape)
    print(T_y)

     
#     print(Tr_x[:,0].reshape(2,1))
#     print(Tr_y[:,0].reshape(1,1))
    