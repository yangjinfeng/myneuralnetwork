# -*- coding: UTF-8 -*-
'''
Created on 2020年7月6日

@author: yangjinfeng
'''
from pandas import read_csv

from yjf.nn.eval import Evaluator
from yjf.nn.layer import Layer, BinaryOutputLayer
from yjf.nn.network import NeuralNet
from yjf.data import preproccess
from sklearn.model_selection import train_test_split
import time

def loadData():
    col_names=["exp_c1","walk_c1","time_pressure_c1","isDisc_c1",\
               "concourse_c1","shift_no_c1","payment_c1","ppp_gdp",\
               "level","h_u_C1","product_involvement_c1","nbuyers_c1",\
               "nstore_crowd_c1","ntravelers_c1","has_c2"]
    
    db = read_csv("file:\\C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\data\\extracted_variable_c12.csv",usecols=col_names)
    Y = db["has_c2"].to_numpy()
    Y = preproccess.labeltransform(Y)
    cate_column= ['isDisc_c1','concourse_c1','shift_no_c1','level','h_u_C1'] 
    X = preproccess.onehot_transform_df(db, cate_column, ["has_c2"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    
    return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = loadData()
    Tr_x = X_train_enc.T
    Tr_y = y_train_enc.reshape(1,y_train_enc.shape[0])
    T_x = X_test_enc.T
    T_y = y_test_enc.reshape(1,y_test_enc.shape[0])

    net = NeuralNet(5000)
    net.setTrainingData(Tr_x, Tr_y)

    net.addLayer(Layer(10,"ReLU",1))   
    net.addLayer(BinaryOutputLayer(1,"sigmoid",1, 0.60))#0.28

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
    result = eval2.eval()
    print(result.toMatrix())
    print("accuracy: "+str(result.accuracy()))
    print("accuracy_p: "+str(result.accuracy_p()))
    print("accuracy_n: "+str(result.accuracy_n()))
    


    print("begin to predict test data... " )
    net.setTestData(T_x, T_y)
    net.test()
    prd = net.getPrediction()
    eval2 = Evaluator(T_y,prd)
    result = eval2.eval()
    print(result.toMatrix())
    print("accuracy: "+str(result.accuracy()))
    print("accuracy_p: "+str(result.accuracy_p()))
    print("accuracy_n: "+str(result.accuracy_n()))



