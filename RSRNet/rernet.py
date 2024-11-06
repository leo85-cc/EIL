#from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Model
# -*- coding: utf-8 -*-
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import backend as K

# from keras import optimizers
# from keras.utils import plot_model#使用plot_mode时打开
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, PReLU, Conv2DTranspose, add, Concatenate, Input, Dropout, BatchNormalization, \
    Activation,GlobalAveragePooling2D,Dense,MaxPooling2D,UpSampling2D,Reshape,multiply
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import regularizers
from tensorflow.keras import regularizers
from tensorflow.keras.activations import sigmoid
from  tensorflow.keras.optimizers import SGD
# model=keras.models.Sequential([
    # # Flatten 变成784 的输入
    # keras.layers.Flatten(input_shape=(28, 28)),
    # # 512的全连接层
    # keras.layers.Dense(512, activation=tf.nn.relu),
    # keras.layers.Dense(1024, activation=tf.nn.relu),
    # # droput 0.2  这个 以后再做讨论 主要是用来防止过拟合
    # keras.layers.Dropout(0.2),
    # # 全连接层 10
    # keras.layers.Dense(10, activation=tf.nn.softmax)])
    # return model

def group_conv(x, filters, kernel, stride, groups):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]  # 计算输入特征图的通道数
    nb_ig = in_channels // groups  # 对输入特征图通道进行分组
    nb_og = filters // groups  # 对输出特征图通道进行分组

    gc_list = []
    for i in range(groups):
        if channel_axis == -1:
            x_group = tf.keras.layers.Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = tf.keras.layers.Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))  # 对每组特征图进行单独卷积

    return Concatenate(axis=channel_axis)(gc_list)  # 在通道上进行特征图的拼接
    return x

def semodule(input):
    xin=input#(input)
    #print("xin.shape")
    #print(xin.shape)
    xg=GlobalAveragePooling2D()(xin)
    #print("xg :",xg[0])
    #print("xg shape:",xg.shape)
    #print("xg shape:",xg.shape[1]//4)
    xl1=Dense(units=xg.shape[1]//4)(xg)
    xl1=Activation('relu')(xl1)
   # print("xl1 shape:", xl1.shape)
    #print("xl1 shape:", xl1.shape[1] // 4)
    xl2=Dense(units=int(xg.shape[1]))(xl1)
    xl2=Activation('sigmoid')(xl2)
    xl2=Reshape((1,1,xg.shape[1]))(xl2)
   # print("xl2 shape:", xl2.shape)

    #xl2=K.expand_dims(xl2,1)
    #print("xl2 shape:", xl2.shape)

    #xl2=K.expand_dims(xl2,2)
    xout=multiply([xl2,xin])

    #print("xl2 shape:", xl2.shape)
    #xout=xin


    #print(xout.shape)

    # print("xg :", xout.shape)
    return xout#(xin,xout)
def resy_pro(input,kernels):
    conv1=Conv2D(filters=kernels,kernel_size=1,strides=1,padding="same",activity_regularizer=regularizers.l1(0.01))(input)
    bn1_1 = BatchNormalization()(conv1, training=False)
    act1_1 = Activation('relu')(bn1_1)
    conv2=group_conv(act1_1,kernels,3,1,16)
    bn2_1 = BatchNormalization()(conv2, training=False)
    act2_1 = Activation('relu')(bn2_1)
    #conv3 = group_conv(act2_1, kernels, 3, 1, 16)
    bn3_1 = BatchNormalization()(act2_1, training=False)
    act3_1 = Activation('relu')(bn3_1)
    conv4=semodule(act3_1)
    merge1 = Concatenate(axis=3)([act3_1, conv4])

    conv5=Conv2D(filters=kernels,kernel_size=1,strides=1,padding="same",activity_regularizer=regularizers.l1(0.01))(merge1)
    merge2 = add([conv1, conv5])
    bn3_1 = BatchNormalization()(merge2, training=False)
    act3_1 = Activation('relu')(bn3_1)
    return act3_1
def resy(input,kernels):
    conv1=Conv2D(filters=kernels,kernel_size=1,strides=1,padding="same",activity_regularizer=regularizers.l1(0.01))(input)
    bn1_1 = BatchNormalization()(conv1, training=False)
    act1_1 = Activation('relu')(bn1_1)

    conv2=group_conv(act1_1,kernels,3,1,16)
    bn2_1 = BatchNormalization()(conv2, training=False)
    act2_1 = Activation('relu')(bn2_1)
    #print("1")

    conv3=semodule(act2_1)
    #print("2")

    merge1 = Concatenate(axis=3)([act2_1, conv3])

    conv4=Conv2D(filters=kernels,kernel_size=1,strides=1,padding="same",activity_regularizer=regularizers.l1(0.01))(merge1)
    merge2 = add([conv1, conv4])
    bn3_1 = BatchNormalization()(merge2, training=False)
    act3_1 = Activation('relu')(bn3_1)
    return act3_1

def stagey(input,layers,kernels):

    for i in range(layers):
       input=resy(input,kernels)
    return input

def stagey_pro(input, layers, kernels):

    for i in range(layers):
        input = resy_pro(input, kernels)
    return input
    #conv2=Conv2D(kernels,1,1,"same",regularizers.l1(0.01))(input)

def resnet(size):
    input = Input((size, size, 3))
    #stem
    conv_stem = Conv2D(filters=32, kernel_size=3, strides=2, padding="same"
                     , activity_regularizer=regularizers.l1(0.01))(input)#stem
    bn1_1 = BatchNormalization()(conv_stem, training=False)
    act1_1 = Activation('relu')(bn1_1)
    pool1=MaxPooling2D(pool_size=(2, 2), strides=2)(act1_1)
    #print("1")
    #bockbone
    conv1=stagey(pool1,1,64)
    #print(conv1.shape)
    pool2=MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    #print("2")
    conv2=stagey(pool2,3,128)
    pool3=MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    #print(conv2.shape)
    #print("3")
    conv3=stagey(pool3,8,320)
    #print(conv3.shape)
    #print("4")
    unpool1=UpSampling2D(size=(2,2))(conv3)
    se1=semodule(unpool1)
    se2=semodule(conv2)
    #print(se1.shape)
    #print(se2.shape)

    merge_1_2 = Concatenate(axis=3)([se1, se2])
    conv4=resy_pro(merge_1_2,128)
    #print(conv4.shape)
    print("5")
    unpool2=UpSampling2D(size=(2,2))(conv4)
    se3=semodule(unpool2)
    se4=semodule(conv1)
    merge_3_4 = Concatenate(axis=3)([se3, se4])
    conv5=resy_pro(merge_3_4,64)

    unpool3 = UpSampling2D(size=(2, 2))(conv5)
    se5 = semodule(unpool3)
    se6 = semodule(conv_stem)
    merge_5_6 = Concatenate(axis=3)([se5, se6])
    conv6 = resy_pro(merge_5_6, 32)

    # print(se3.shape)
    # print(se4.shape)
    # print(se5.shape)
    # print(se6.shape)
    unpool4 = UpSampling2D(size=(2, 2))(conv6)

    Routput = Conv2D(filters=2, kernel_size=1,strides=1, padding='same', activation='softmax')(unpool4)
    o =Dense(units=2, activation='softmax')(Routput)
    model = Model(input,o)
    return model

if __name__ =="__main__":


    resmodel = resnet()
    #Model.summary(semodule((128,128,3)))
    Model.summary(resmodel)

