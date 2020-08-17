# -*- coding: UTF-8 -*-
'''
Created on 2020年8月5日

@author: yangjinfeng
'''

'''
mat(input): data_format='channels_last'    
        "NHWC": [batch, height, width, channels].
        "NCHW": [batch, channels, height, width].
        
fs(filter): A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
'''
import numpy as np
from tensorflow.python.ops import nn_ops
import tensorflow as tf
import h5py

def load_dataset():
    train_dataset = h5py.File('C:/Users/admin/mycnn/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('C:/Users/admin/mycnn/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def convolve(cover,f):
    ele = cover * f
    return np.sum(ele)


def getPadAndOutShape(mat_shape,filter_size,stride,padding):
    outshape = None
    pad_w = None
    pad_h = None
    h = mat_shape[0]
    w = mat_shape[1]
    if padding == "SAME":
        out_h = int(np.ceil(float(h)/float(stride)))
        out_w = int(np.ceil(float(w)/float(stride)))
        outshape = (out_h,out_w)
        pad_h_total= stride * (out_h - 1) - (h - filter_size)
        pad_h_top = int(pad_h_total / 2)
        pad_h_bottom = pad_h_total - pad_h_top        
        pad_w_total= stride * (out_w - 1) - (w - filter_size)
        pad_w_left = int(pad_w_total / 2)
        pad_w_right = pad_w_total - pad_w_left
        pad_h = (pad_h_top,pad_h_bottom)
        pad_w = (pad_w_left,pad_w_right)
    else:
        out_h = int(np.ceil(float(h - filter_size + 1)/float(stride)))
        out_w = int(np.ceil(float(w - filter_size + 1)/float(stride)))
        outshape = (out_h,out_w)
        pad_h = (0,0)
        pad_w = (0,0)
    return outshape,pad_h,pad_w

# mats: [batch, height, width, channels]
def mypad(mats,pad_h,pad_w):
    shp = mats.shape
    h = shp[1]+pad_h[0]+pad_h[1]
    w = shp[2]+pad_w[0]+pad_w[1]
    new_shape = (shp[0],h,w,shp[3])
    padmats = np.zeros(new_shape,dtype=float)
    padmats[:, pad_h[0]:h-pad_h[1], pad_w[0]:w-pad_w[1],:] = mats
    return padmats

# mat : 一个样本 [height, width, channels]
#f1是一个过滤器,h,w,c
def conv_f1_forward(mat,f1,stride=1,padding="VALID"):
    f = f1.shape[0] 
    out_shape,pad_h,pad_w = getPadAndOutShape(mat.shape,f1.shape[0],stride,padding)
    h = out_shape[0]
    w = out_shape[1]
    mat2 = np.pad(mat,((pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0)) 
    result = np.zeros((h,w),dtype=float);   
    for i in range(h):
        i_s = i * stride
        for j in range(w):
            j_s = j * stride
            cover = mat2[i_s:i_s + f,j_s:j_s + f,:]
            result[i,j]=convolve(cover,f1)
    return result

# mat : 一个样本 [height, width, channels]
# padding="VALID" || "same"
# fs是一组过滤器，h,w,c,outchannels
# fs[i]是一个三位的过滤，第三维的通道数与mat的通道数相同
def conv_one_forward2(mat,fs,stride=1,padding="VALID"):
    result = None
    outchannles = fs.shape[3]
    for i in range(outchannles):
        result_i = conv_f1_forward(mat,fs[:,:,:,i],stride,padding)
        if result is None:
            result = result_i
        else:
            result = np.dstack((result,result_i))
    return result

# mat : 一个样本 [height, width, channels]
# padding="VALID" || "SAME"
# fs是一组过滤器，(h,w,c,outchannels),fs[:,:,:,i]是一个三位的过滤，第三维的通道数与mat的通道数相同
def conv_one_forward(mat,fs,stride=1,padding="VALID"):
    f = fs.shape[0] 
    
    out_shape,pad_h,pad_w = getPadAndOutShape(mat.shape,fs.shape[0],stride,padding)
    h = out_shape[0]
    w = out_shape[1]
    mat2 = np.pad(mat,((pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0)) 
    
    channels = fs.shape[3]
    result = np.zeros((h,w,channels),dtype=float);
    for c in range(channels):
        conved = result[:,:,c]
        for i in range(h):
            i_s = i * stride
            for j in range(w):
                j_s = j * stride
                cover = mat2[i_s:i_s + f,j_s:j_s + f,:]
                conved[i,j]=convolve(cover,fs[:,:,:,c])
    return result

# mats: [batch, height, width, channels]
# padding="VALID" || "SAME"
# fs是一组过滤器，fs[:,:,i]是一个三位的过滤，第三维的通道数与mat的通道数相同
def conv_forward(mats,fs,stride=1,padding="VALID"):
    m = len(mats)
    z_data = []
    for i in range(m):
        z_data.append(conv_one_forward(mats[i], fs, stride, padding))
#         z_data.append(conv_one_forward2(mats[i], fs, stride, padding))
    return np.array(z_data)

'''
D:\myapplication\anaconda3\Lib\site-packages\tensorflow\python\ops\nn_ops.py

mat(input): data_format='channels_last'    
        "NHWC": [batch, height, width, channels].
        "NCHW": [batch, channels, height, width].
fs(filter): A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
      
stride: Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
        horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
        
  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.        
     
     about right-multiplies:
     https://www.quantstart.com/articles/matrix-inversion-linear-algebra-for-deep-learning-part-3/
     
  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q]
                          * filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
'''
def conv_forward_tf(mat,fs,stride=1,padding="VALID"):
    strd = [1,stride,stride,1]
    out = nn_ops.conv2d(mat.astype(np.float64),fs,strd,padding,data_format="NHWC")
    return np.array(out)

    
#method : MAX, AVG
def pool(mat2d,method="MAX"):
    if method == "MAX":
        return np.max(mat2d)
    else:
        return np.mean(mat2d)

#mat2d是单通道矩阵
def pool_f1_forward(mat2d,fsize,stride=1,padding="VALID",method="MAX"):
    f = fsize

    out_shape,pad_h,pad_w = getPadAndOutShape(mat2d.shape,fsize,stride,padding)
    h = out_shape[0]
    w = out_shape[1]
    mat2 = np.pad(mat,((pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0)) 
    
    
    result = np.zeros((h,w),dtype=float);
    for i in range(h):
        i_s = i * stride
        for j in range(w):
            j_s = j * stride
            cover = mat2[i_s:i_s + f,j_s:j_s + f]
            result[i,j]=pool(cover,method)
    return result

#mat是多通道矩阵，三维，一个样本
def pool_one_forward2(mat,fsize,stride=1,padding="VALID",method="MAX"):
    result = None
    (_,_,channels)= mat.shape
    for i in range(channels):
        result_i = pool_f1_forward(mat[:,:,i],fsize,stride,padding,method)
        if result is None:
            result = result_i
        else:
            result = np.dstack((result,result_i))
    return result
    
#mat是多通道矩阵，三维，一个样本
def pool_one_forward(mat,fsize,stride=1,padding="VALID",method="MAX"):
    f = fsize
    out_shape,pad_h,pad_w = getPadAndOutShape(mat.shape,fsize,stride,padding)
    h = out_shape[0]
    w = out_shape[1]
    mat2 = np.pad(mat,((pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0)) 
    
    channels = mat.shape[2]
    result = np.zeros((h,w,channels),dtype=float);
    for c in range(channels):
        pooled = result[:,:,c]
        for i in range(h):
            i_s = i * stride
            for j in range(w):
                j_s = j * stride
                cover = mat2[i_s:i_s + f,j_s:j_s + f,c]
                pooled[i,j]=pool(cover,method)
    return result


#mats是多个样本多通道矩阵，四维，
#"NHWC": [batch, height, width, channels].
def pool_forward(mats,fsize,stride=1,padding="VALID",method="MAX"):
    m = len(mats)
    z_data = []
    for i in range(m):
        z_data.append(pool_one_forward(mats[i],fsize,stride,padding,method))
    return np.array(z_data)    

#mats是多个样本多通道矩阵，四维，
#"NHWC": [batch, height, width, channels].
def pool_forward_tf(mats,fsize,stride=1,padding="VALID",method="MAX"):

    strd = [stride,stride]
    poolshape = [fsize,fsize]
    out = nn_ops.pool(mats.astype(np.float64),\
                      poolshape,\
                      method,\
                      padding=padding,\
                      strides=strd)
    
#     strd = [1,stride,stride,1]
#     poolshape = [1,fsize,fsize,1]
#     out = tf.nn.max_pool(mats,poolshape,strd,padding,)
    return np.array(out)

'''
根据当前层的dZ计算上一层的dA

input_sizes(input): data_format='channels_last'    
        "NHWC": [batch, height, width, channels].
        "NCHW": [batch, channels, height, width].
        上一层的A就是input，与dA形状相同
filters(filter): A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
out_backprop
     当前层的dZ，是激活函数的梯度 
    "NHWC": [batch, height, width, channels].
stride: 1或者2........
返回dA
'''
def conv_backprop_input(input_sizes,filters,out_backprop,stride,padding='VALID'):
    _,pad_h,pad_w = getPadAndOutShape(input_sizes[1:],filters.shape[0],stride,padding)
    
    f = filters.shape[0]
    dZ = out_backprop
    dA = np.zeros(input_sizes,dtype=float)
#     dA2 = mypad(dA,(pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]))
    dA2 = np.pad(dA,((0,0),(pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0))
    for m in range(input_sizes[0]): #样本数
        for c in range(dZ.shape[3]): #dZ的通道数
            for i in range(dZ.shape[1]):
                i_s = i * stride
                for j in range(dZ.shape[2]):
                    j_s = j * stride
                    dA2[m, i_s:i_s+f, j_s:j_s+f] = dA2[m, i_s:i_s+f, j_s:j_s+f] + dZ[m,i,j,c] * filters[:,:,:,c]
                    
    (_,h,w,_) = dA2.shape
    return dA2[:,pad_h[0]:h-pad_h[1],pad_w[0]:w-pad_w[1],:]


def conv_backprop_input_tf(input_sizes,filters,out_backprop,stride,padding='VALID'):
    strd = [1,stride,stride,1]
    d_A_backprop_input = nn_ops.conv2d_backprop_input(input_sizes=input_sizes,
                                                     filter=filters,
                                                     out_backprop=out_backprop,
                                                     strides=strd,
                                                     padding=padding)
    return np.array(d_A_backprop_input)

'''
根据当前层的dZ计算上一层的dA

inputmats(input): data_format='channels_last'    
        "NHWC": [batch, height, width, channels].
        "NCHW": [batch, channels, height, width].
        上一层的A就是input，
filter_sizes: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
out_backprop
     当前层的dZ，是激活函数的梯度 
    "NHWC": [batch, height, width, channels].
stride: 1或者2.......
返回dW
'''
def conv_backprop_filter(inputmats,filter_sizes,out_backprop,stride,padding='VALID'):
    _,pad_h,pad_w = getPadAndOutShape(inputmats.shape[1:],filter_sizes[0],stride,padding)
    
    preA = inputmats
    f = filter_sizes[0]
    dZ = out_backprop
    dW = np.zeros(filter_sizes,dtype=float)
    preA2 = np.pad(preA,((0,0),(pad_h[0],pad_h[1]),(pad_w[0],pad_w[1]),(0,0)),'constant',constant_values=(0,0))
    
    for m in range(preA2.shape[0]): #样本数
        for c in range(dZ.shape[3]): #dZ的通道数
            for i in range(dZ.shape[1]):
                i_s = i * stride
                for j in range(dZ.shape[2]):
                    j_s = j * stride
                    dW[:,:,:,c] = dW[:,:,:,c] + dZ[m,i,j,c] *  preA2[m, i_s:i_s+f, j_s:j_s+f]

    return dW/preA2.shape[0]


def conv_backprop_filter_tf(inputmats,filter_sizes,out_backprop,stride,padding='VALID'):
    strd = [1,stride,stride,1]
    d_w_backprop_filter = nn_ops.conv2d_backprop_filter(input=inputmats,
                                                   filter_sizes=filter_sizes,
                                                   out_backprop=out_backprop,
                                                   strides=strd,
                                                   padding=padding)
    return np.array(d_w_backprop_filter)
    

if __name__ == '__main__':
    #构造两个三通道卷积核
    f1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ff = np.dstack((f1,f1))
    filter1 = np.dstack((ff,f1))
    
    filter2 = np.zeros((3,3,3),dtype=float)
    f2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    filter2[:,:,0]=f2
    filter2[:,:,1]=f2
    filter2[:,:,2]=f2
    
    fs = np.zeros((3,3,3,2),dtype=float)
    fs[:,:,:,0] = filter1
    fs[:,:,:,1] = filter2
#     print(fs.shape)
    
    np.random.seed(1)
    mats = np.array([np.random.randint(0,2,size=(8,8,3))])*1.0
#     print(mats.shape)
    
    result = conv_forward(mats,fs,stride=2,padding="SAME")
    # print(result.shape)
    # print(result[0,:,:,0])
    # print(result[0,:,:,1])
    
    outshape = result.shape
    np.random.seed(1)
    d_out = np.random.rand(outshape[0],outshape[1],outshape[2],outshape[3])* 0.1
    # print(d_out)
#     print(d_out.shape)

#     dA = conv_backprop_input(input_sizes=mats.shape,filters=fs, out_backprop=d_out,stride=2,padding='SAME')
#     print(dA)
#     dA = conv_backprop_input_tf(input_sizes=mats.shape,filters=fs, out_backprop=d_out,stride=2,padding='SAME')
#     print(dA)
    
    dW = conv_backprop_filter(inputmats=mats,filter_sizes=fs.shape,out_backprop=d_out,stride=2,padding='SAME')
    print(dW)
    dW = conv_backprop_filter_tf(inputmats=mats,filter_sizes=fs.shape,out_backprop=d_out,stride=2,padding='SAME')
    print(dW)

if __name__ == '__main2__':
    #构造三通道卷积核
    f1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ff = np.dstack((f1,f1))
    filter1 = np.dstack((ff,f1))
    filter2 = np.zeros((3,3,3),dtype=float)
    f2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    filter2[:,:,0]=f2
    filter2[:,:,1]=f2
    filter2[:,:,2]=f2
    fs = np.zeros((3,3,3,2),dtype=float)
    fs[:,:,:,0] = filter1
    fs[:,:,:,1] = filter2
    
    np.random.seed(1)
    mat = np.array([np.random.randint(0,2,size=(8,8,3))])
    
    result = conv_forward(mat,fs,padding="VALID")
    print(result.shape)
    print(result[0,:,:,0])
    print(result[0,:,:,1])
#     
    
    pooledresult2 = pool_forward_tf(result,3,3,padding="SAME",method="AVG")
#     pooledresult2 = tf.nn.max_pool(result,3,3,"SAME")
    print(pooledresult2.shape)
    print(pooledresult2[0,:,:,0])
    print(pooledresult2[0,:,:,1])

#     max = tf.nn.max_pool
    print(result.shape)
    print(result[0,:,:,0])
    print(result[0,:,:,1]) 
    
    pooledresult = pool_forward(result,3,3,padding="SAME",method="AVG")
    print(pooledresult.shape)
    print(pooledresult[0,:,:,0])
    print(pooledresult[0,:,:,1])
    
if __name__ == '__main1__':
    tr_x,tr_y,t_x,t_y,classes = load_dataset()
    f1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ff = np.dstack((f1,f1))
    filter1 = np.dstack((ff,f1))
    
    filter2 = np.zeros((3,3,3),dtype=float)
    f2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    filter2[:,:,0]=f2
    filter2[:,:,1]=f2
    filter2[:,:,2]=f2
    
    fs = np.zeros((3,3,3,2),dtype=float)
    fs[:,:,:,0] = f1
    fs[:,:,:,1] = f2
#     out = conv_forward_tf(tr_x,fs,1,padding="VALID")
    out = conv_forward(tr_x,fs,3,padding="SAME")
    out = np.array(out).astype(np.int16)
#     out = nn_ops.conv2d(tr_x.astype(np.float64),fs,[1,1,1,1],padding="VALID",data_format="NHWC")
    print(out.shape)
    print(out[0,:,:,0])
    
    out2 = conv_forward_tf(tr_x,fs,3,padding="SAME")
    out2 = np.array(out2).astype(np.int16)
    print(out2.shape)
    print(out2[0,:,:,0])
    
if __name__ == '__main2__':
    xx = tf.nn.max_pool
    mat = np.array([\
               [1,1,1,1,1,1,1,1],\
              [10,2,2,2,2,2,12,14],\
               [3,3,3,3,5,6,3,3],\
               [4,4,4,4,4,4,4,4],\
               [5,5,5,5,5,5,5,5],\
               [6,6,6,6,6,6,6,6],\
               [7,7,9,7,7,7,7,7],\
               [8,8,8,8,8,8,8,8]])
    mat = mat.reshape(1,8,8,1)
#     out = xx(mat,(1,7,7,1),(1,2,2,1),"SAME")
    out = pool_forward_tf(mat,5,3,padding="SAME",method="MAX")
    print(out[:,:,:,0])
    out2 = pool_forward(mat,5,3,padding="SAME",method="MAX")
    print(out2[:,:,:,0])