#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-05 23:05:15
# @Author  : Yang Jinfeng (${email})

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

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



def prepare_data(x,y):
    # oe = OrdinalEncoder()
    ohe = OneHotEncoder()
    ohe.fit(x)
    X_enc = ohe.transform(x).A
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
    print(X_train_enc.shape)
    # define the  model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=0)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))
    print(_)


if __name__ == '__main__':
    # pass
    test()