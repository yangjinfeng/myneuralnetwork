# -*- coding: UTF-8 -*-
'''
Created on 2020年8月3日

@author: yangjinfeng
'''
import numpy as np

def test_encode_decode():
    str = "哈尔滨"
    str2 = "黑龙江"
    bs = str.encode("gbk")
    bs2 = str2.encode("gbk")
    print(bs)
    print(len(bs))
    print(type(bs))
    str2 = bs.decode("gbk")
    print(str2)
    print(bs[0:2].decode("gbk"))
    bsa = np.array([bs,bs2])
    print(type(bsa))



if __name__ == '__main__':
    test_encode_decode()