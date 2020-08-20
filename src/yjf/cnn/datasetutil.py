# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''

import numpy as np
import h5py


    
# data_format='channels_last'    
def load_dataset():
    train_dataset = h5py.File('C:/Users/admin/mycnn/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
#     train_set_x_orig = np.array(train_dataset["train_set_x"][0:10]) # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][0:10]) # your train set labels


    test_dataset = h5py.File('C:/Users/admin/mycnn/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


#tr_x.shape: (209, 64, 64, 3)
#tr_y.shape: (1, 209)
#t_x.shape: (50, 64, 64, 3)
#t_x.shape: (1, 50)
def loadData1():
    tr_x,tr_y,t_x,t_y,classes = load_dataset()
    return tr_x,tr_y


if __name__ == '__main__':
    tr_x,tr_y = loadData1()
    print(tr_x.shape)