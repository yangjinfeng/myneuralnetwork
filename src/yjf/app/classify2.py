#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-05 23:05:15
# @Author  : Yang Jinfeng (${email})

import time

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from yjf.nn.eval import Evaluator
from yjf.nn.layer import Layer, BinaryOutputLayer
from yjf.nn.network import NeuralNet


filename="C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\data\\breast-cancer.data"

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
	X = X.astype(str)
	# reshape target to be a 2d array
	y = y.reshape((len(y), 1))
	return X, y

# x,y=load_dataset(filename)
# oe = OrdinalEncoder()
# oe.fit(x)
# X_train_enc = oe.transform(x)
# print(X_train_enc.shape)
# print(X_train_enc[0])
# le = LabelEncoder()
# le.fit(y)
# y_train_enc = le.transform(y)
# print(y_train_enc.shape)
# print(y_train_enc)


def prepare_data(x,y):
	# oe = OrdinalEncoder()
	oe = OneHotEncoder()
	oe.fit(x)
	X_enc = oe.transform(x).A
	le = LabelEncoder()
	le.fit(y)
	y_enc = le.transform(y)

	X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.33, random_state=1)
	return X_train, X_test, y_train, y_test


def test():
	# load the dataset
	X, y = load_dataset(filename)
	# split into train and test sets
	X_train_enc, X_test_enc, y_train_enc, y_test_enc = prepare_data(X, y)
	Tr_x = X_train_enc.T
	Tr_y = y_train_enc.reshape(1,y_train_enc.shape[0])
	T_x = X_test_enc.T
	T_y = y_test_enc.reshape(1,y_test_enc.shape[0])
	# print(Tr_x.shape)
	# print(Tr_y.shape)
	# print(T_x.shape)
	# print(T_y.shape)

	# print(T_x)
	# print(T_y)
	net = NeuralNet(5000)
	# net.setMiniBatch(False)
	net.setTrainingData(Tr_x, Tr_y)


	net.addLayer(Layer(10,"ReLU",1))   
	net.addLayer(BinaryOutputLayer(1,"sigmoid",1,0.5))


	print("begin to train ... " )
	tic = time.time()
	net.train()

	toc = time.time()
	print("time lasted: " + str(1000*(toc-tic)))

	print("begin to predict training data... " )
	net.setTestData(Tr_x, Tr_y)
	net.test()
	prd = net.getPrediction()
	eval2 = Evaluator(Tr_y,prd)
	print(eval2.eval().accuracy())


	print("begin to predict test data... " )
	net.setTestData(T_x, T_y)
	net.test()
	prd = net.getPrediction()
	eval2 = Evaluator(T_y,prd)
	print(eval2.eval().accuracy())

if __name__ == '__main__':
	# pass
	test()