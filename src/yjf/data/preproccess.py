# -*- coding: UTF-8 -*-
'''
Created on 2020年7月6日

@author: yangjinfeng
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder


def get_df_locs(dataframe,col_names):
    index=[]
    for name in col_names:
        index.append(dataframe.columns.get_loc(name))
    return index
'''
a=np.array([[1.2,10,"A"],
[0.9,12,"A"],[1.3,9,"B"],[3,1.1,"C"],[4,1.1,"A"],[9,1.4,"B"]])
data  m,n    m是样本数，n是特征数
'''
def onehot_transform(data,cate_cindex_array):
    #先创建一个空的矩阵，也可以这样创建：a=np.array([[],[]])
    transformed = np.empty((data.shape[0], 0)) 
    oh = OneHotEncoder()
    for i in range(data.shape[1]):   
        if (i in cate_cindex_array):
            ohc=oh.fit_transform(data[:,i].reshape(-1,1)).A
            transformed = np.hstack((transformed,ohc))
        else:
            transformed = np.hstack((transformed,data[:,i].reshape(-1,1)))
    return  transformed           

'''
data  n,m    m是样本数，n是特征数
'''
def onehot_transform2(data,cate_cindex_array):
    data = data.T
    transformed = onehot_transform(data,cate_cindex_array)
    return  transformed.T   

def labeltransform(data):
    le= LabelEncoder()
    lec = le.fit_transform(data.reshape(-1,1))
    return lec

def onehot_transform_df(dataframe,cate_colname_array,excludes):
    transformed = np.empty((dataframe.shape[0], 0)) 
    oh = OneHotEncoder()
    for name in dataframe.columns:
        if name in excludes:
            continue
        col = dataframe[name].to_numpy().reshape(-1,1)
        if name in cate_colname_array:
            col=oh.fit_transform(col).A
        transformed = np.hstack((transformed,col))
    return  transformed  
            
if __name__ == '__main__':     
    
#     a=np.array([[1.2,10,"A"],\
#     [0.9,12,"A"],[1.3,9,"B"],[3,1.1,"C"],[4,1.1,"A"],[9,1.4,"B"]])
#     b = onehot_transform2(a.T, [2]) 
#     print(b)
    y=np.array(["True","False"]).reshape(2,1)
    print(labeltransform(y,0))
    