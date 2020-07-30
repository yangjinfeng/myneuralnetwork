#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 12:39:59
# @Author  : Yang Jinfeng (${email})
# @Link    : ${link}
# @Version : $Id$

import sklearn
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

db = pd.read_csv("C:\\Users\\yangjinfeng\\git\\myneuralnetwork\\data\\extracted_variable_c12.csv")
c1=db["concourse_c1"]
#print(c1.describe())
print(type(c1))

le = LabelEncoder()
letrans = le.fit_transform(c1)
print(type(letrans))

oh = OneHotEncoder()
ohc=oh.fit_transform(letrans.reshape(-1, 1))
print(ohc.shape)
ohca=ohc.A
print(ohca[0:3])
print("112");