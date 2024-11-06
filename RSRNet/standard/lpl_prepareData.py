'''18/08/22 lpl 16:28
This script aims to preprocess the data for model
Update1:19/05/31，添加注释
Basic
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2 as cv
import lmdb
import numpy as np

import auxiliary
import auxiliary_3C
import auxiliary_1C
'''对文件块的名字进行拆分，主要拆分为两个原始文件名和当前块的编号'''
def split_str(name_index):
    '''
    :param name_index: 文件名列表
    :return:原始文件名(name)和当前块编号(index)
    '''
    '''split the name_index to name and index'''
    name_index_split = bytes.decode(name_index).split('_')
    name=name_index_split[0]
    for i in range(1,len(name_index_split)-1):
        name=name+'_'+name_index_split[i]
    index=int(name_index_split[-1])
    return name,index
'''根据name_list对影像块进行合并，得到原始未分割的图像'''
def merge_image(image_list,name_list):
    '''
    :param image_list:原始影像所有的块
    :param name_list: 原始影像的名字以及索引号
    :return:
    '''
    '''Merge image.'''
    if len(image_list)!=len(name_list):
        print('The number of image is not equal to name list.' )
        sys.exit(0)
    n=(len(image_list)) ** 0.5
    if int(n)!=n:
        print('The number of image is illegal.')
        sys.exit(0)
    shape=image_list[0].shape
    image=np.zeros((int(shape[0]*n),int(shape[1]*n)), dtype=np.uint8)
    for i in range(0, len(name_list)):
        pos=name_list[i][1]
        pos_row=pos-int(pos/n)*n
        pos_row=int(pos_row)
        pos_column=int(pos/n)
        start_row=pos_row*shape[0];end_row=(pos_row+1)*shape[0]
        start_column = pos_column * shape[1];end_column = (pos_column+1)*shape[1]
        image[start_row:end_row,start_column:end_column]=np.reshape(image_list[i],(shape[0],shape[1]))
        #print(pos_row,pos_column)
    return image,name_list[0][0]
'''按照名字，对预测块进行分类，为影像块的合并做准备。'''
def merge_image_list(predicted_list,name_list):
    '''Aiming to merge the predicted result of model to images list.'''
    if len(predicted_list)!=len(name_list) or len(name_list)==0  \
        or len(predicted_list)==0:
        print('Predicted results are errors!')
        return
    image_list=[]
    merge=[]
    image_name_list=[]
    merge_name=[]
    for index in range(0, len(predicted_list)):
        if index==0:
            map_patch = predicted_list[index]
            cur_name, cur_index = split_str(name_list[index])
            merge.append(map_patch)
            merge_name.append((cur_name,cur_index))
        else:
            map_patch=predicted_list[index]
            next_name,next_index=split_str(name_list[index])
            if cur_name == next_name:
                merge.append(map_patch)
                merge_name.append((next_name, next_index))
            else:
                image_list.append(merge)
                image_name_list.append(merge_name)
                cur_name=next_name
                merge=[]
                merge_name=[]
                merge.append(map_patch)
                merge_name.append((cur_name, cur_index))
    if index==len(predicted_list)-1 and cur_name==next_name:
        image_list.append(merge)
        image_name_list.append(merge_name)
    return image_list, image_name_list
#拼接影像
def merge_map_segmentation(seg,name_list,class_num=2):
    seg_inverse = []
    images=[];names=[]
    for i in range(0, len(seg)):
        #将概率map转化为（0,1）矩阵
        shape=seg[i].shape
        if len(shape)==4:
            seg[i]=np.reshape(seg[i], [shape[1], shape[2],shape[3]])
        if class_num==2:
            inverse = auxiliary.convert_labels2C_inverse(seg[i])
        elif class_num==3:
            inverse = auxiliary_3C.labels_inverse_3C(seg[i])
        seg_inverse.append(inverse)
    image_seg, name_seg = merge_image_list(seg_inverse, name_list)
    for i in range(0, len(image_seg)):
        image, name = merge_image(image_seg[i], name_seg[i])
        images.append(image);names.append(name)
    return images,names
'''将合并后的影像存入本地(path)'''
def save_image(image,path):
    '''save image to output_dir'''
    cv.imwrite(path,image)
'''将预测结果，按照名字进行合并，并将合并结果存入本地'''
def save_results(predicted_results,name_list,path_root):
    predicted_results_inverse=[]
    for i in range (0,len(predicted_results)):
        label_inverse = []
        for j in range(0,len(predicted_results[i])):
            inverse=\
                auxiliary.convert_labels2C_inverse(predicted_results[i][j])
            label_inverse.append(inverse)
        predicted_results_inverse.append(label_inverse)
    image_list,name_list=merge_image_list(predicted_results_inverse, name_list)
    for i in range(0,len(image_list)):
        image,name=merge_image(image_list[i],name_list[i])
        path = os.path.join(path_root, auxiliary.generate_name_time())
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, name + '.tif')
        save_image(image,path)

def Create_minibatch(args,o_cur, l_cur):
    batchs_data=[]
    batchs_label=[]
    key=[]
    x_minibatch = []
    y_minibatch = []
    key_minibatch=[]
    i = 0
    while True:
        o_key, o_val = o_cur.item()
        l_key, l_val = l_cur.item()
        if o_key != l_key:
            raise ValueError(
                'Keys of ortho and label patches are different: '
                '{} != {}'.format(o_key, l_key))

        # prepare patch
        o_side = args.sat_size
        l_side = args.map_size
        if args.map_channel==1:
            #print("map_channel==1")
            o_patch = np.fromstring(
                o_val, dtype=np.uint8).reshape((o_side, o_side, 3))
            l_patch = np.fromstring(
                l_val, dtype=np.uint8).reshape((l_side, l_side, args.map_channel))
        else:
            o_patch = np.fromstring(
                o_val, dtype=np.float16).reshape((o_side, o_side, 3))
            l_patch = np.fromstring(
                l_val, dtype=np.uint8).reshape((l_side, l_side, args.map_channel))
        # add patch
        x_minibatch.append(o_patch)
        y_minibatch.append(l_patch)
        key_minibatch.append(o_key)
        o_ret = o_cur.next()
        l_ret = l_cur.next()
        if ((not o_ret) and (not l_ret)) or len(x_minibatch) ==args.batchsize:
            if args.map_channel==1:
                x_minibatch = np.asarray(x_minibatch, dtype=np.uint8)
                y_minibatch = np.asarray(y_minibatch, dtype=np.uint8)
            else:
                x_minibatch = np.asarray(x_minibatch, dtype=np.float16)
                y_minibatch = np.asarray(y_minibatch, dtype=np.uint8)
            batchs_data.append(x_minibatch)
            batchs_label.append(y_minibatch)
            key.append(key_minibatch)
            i += len(x_minibatch)
            x_minibatch = []
            y_minibatch = []
            key_minibatch=[]
        if ((not o_ret) and (not l_ret)):
            break
    return batchs_data,batchs_label,key
def Create_minibatch_mul_gpu(args,o_cur, l_cur,num):
    batchs_data=[]
    batchs_label=[]
    key=[]
    x_minibatch = []
    y_minibatch = []
    key_minibatch=[]
    i = 0
    while True:
        o_key, o_val = o_cur.item()
        l_key, l_val = l_cur.item()
        if o_key != l_key:
            raise ValueError(
                'Keys of ortho and label patches are different: '
                '{} != {}'.format(o_key, l_key))

        # prepare patch
        o_side = args.sat_size
        l_side = args.map_size
        if args.map_channel==1:
            o_patch = np.fromstring(
                o_val, dtype=np.uint8).reshape((o_side, o_side, 3))
            l_patch = np.fromstring(
                l_val, dtype=np.uint8).reshape((l_side, l_side, args.map_channel))
        else:
            o_patch = np.fromstring(
                o_val, dtype=np.float16).reshape((o_side, o_side, 3))
            l_patch = np.fromstring(
                l_val, dtype=np.uint8).reshape((l_side, l_side, args.map_channel))
        # add patch
        x_minibatch.append(o_patch)
        y_minibatch.append(l_patch)
        key_minibatch.append(o_key)
        o_ret = o_cur.next()
        l_ret = l_cur.next()
        if ((not o_ret) and (not l_ret)) or len(x_minibatch) ==args.batchsize*num:
            if args.map_channel==1:
                x_minibatch = np.asarray(x_minibatch, dtype=np.uint8)
                y_minibatch = np.asarray(y_minibatch, dtype=np.uint8)
            else:
                x_minibatch = np.asarray(x_minibatch, dtype=np.float16)
                y_minibatch = np.asarray(y_minibatch, dtype=np.uint8)
            batchs_data.append(x_minibatch)
            batchs_label.append(y_minibatch)
            key.append(key_minibatch)
            i += len(x_minibatch)
            x_minibatch = []
            y_minibatch = []
            key_minibatch=[]
        if ((not o_ret) and (not l_ret)):
            break
    return  batchs_data,batchs_label,key
def Get_cursor(db_fn):
    while True:
      try:
        env = lmdb.Environment(db_fn, readonly=True, max_readers=256)
        break
      except:
        continue
    #env = lmdb.open(db_fn)
    txn = env.begin(write=False, buffers=False)
    cur = txn.cursor()
    cur.next()
    return cur, txn, env.stat()['entries']
'''对lmdb文件进行读取'''
def GetData(args,image_db,label_db):
    o_cur, o_txn, args.N = Get_cursor(image_db)
    l_cur, l_txn, _ = Get_cursor(label_db)
    #get the training data from the lmdb file
    batchs_data,batchs_label, key=Create_minibatch(args, o_cur, l_cur)
    return batchs_data,batchs_label,key
def GetData_mul_gpu(args,num,image_db,label_db):
    num = num
    o_cur, o_txn, args.N = Get_cursor(image_db)
    l_cur, l_txn, _ = Get_cursor(label_db)
    #get the training data from the lmdb file
    batchs_data,batchs_label,key=Create_minibatch_mul_gpu(args,o_cur, l_cur,num)
    return batchs_data,batchs_label,key
def Create_args():
    parser = argparse.ArgumentParser()
    # Training settings
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--sat_size', type=int, default=92)
    parser.add_argument('--map_size', type=int, default=8)
    parser.add_argument('--test_image_db', type=str, default=
    '/data1/lpl/ssai-lpl/roads/test_roads_92_8_8_00_40_regular/lmdb/test_sat')
    parser.add_argument('--test_label_db', type=str, default=
    '/data1/lpl/ssai-lpl/roads/test_roads_92_8_8_00_40_regular/lmdb/test_map')
    args = parser.parse_args()
    return args
'''测试函数，请勿使用'''






