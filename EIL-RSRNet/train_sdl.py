# coding=utf-8
'''''
19/07/03 lpl 15:33
FCN模型，输入数据为24*24*3的遥感影像片，输出为24*24*2的标签地图
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import cmath

sys.path.append("./standard")
import argparse
import tensorflow as tf
import numpy as np
import os
import time
# import lpl_prepareData_1C as lpl_prepareData
import lpl_accuracy as acc
import lpl_prepareData
import lpl_logger
import lpl_tensorboard
import lpl_accuracy as acc
import socket
import rernet as NN
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import models
# from segnet_quick import build


from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD


# 创建一个日志记录器s


# 定义参数
def Create_args():
    parser = argparse.ArgumentParser()
    # Training settings
    size = '128'
    # T='6'
    a = int(size)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--wro_weight', type=int, default=4)
    parser.add_argument('--sat_size', type=int, default=a)
    parser.add_argument('--sat_channel', type=int, default=3)
    parser.add_argument('--map_size', type=int, default=a)
    parser.add_argument('--map_channel', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--gpu_index', type=str, default="0")
    parser.add_argument('--gpu_rate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-7)  # 设置学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)  # 设置衰减率
    parser.add_argument('--decay_steps', type=float, default=100000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer_', type=str, default="sgd")#sgd adam adagrad adadelta
    parser.add_argument('--image_db', type=str, default=

                        #'/public/home/zzuegr01/sdj/cheng_data/train_cheng_photo_crop/sat_s')
                        #'/public/home/zzuegr01/ljm/data_utils/huijidata/0405/osm_train_128_s/sat')
                        '/public/home/zzuegr01/zt/lpl_dataset/train/sat_s')

    parser.add_argument('--label_db', type=str, default=

                        #'/public/home/zzuegr01/ljm/data_utils/huijidata/0405/osm_train_0.1_128_map_s')
                        '/public/home/zzuegr01/zt/lpl_dataset/train/map_s')
    

    parser.add_argument('--test_image_db', type=str, default=
                        #'/public/home/zzuegr01/ljm/data_utils/huijidata/0405/osm_test_128_s/sat')
                        '/public/home/zzuegr01/zt/lpl_dataset/test/sat_s')
                        

    parser.add_argument('--test_label_db', type=str, default=
                       # '/public/home/zzuegr01/lpl/source_datasets/mnih/mass_roads/pieces_384/210517_128_s/test/map')
                       #'/public/home/zzuegr01/ljm/data_utils/huijidata/0405/osm_test_128_s/map')
                        '/public/home/zzuegr01/zt/lpl_dataset/test/map_s')

    parser.add_argument('--image_save', type=str, default='./outdata/zim/image_save/')
    parser.add_argument('--save_path', type=str, default='./outdata/zim/model_save/')
    parser.add_argument('--log_path', type=str, default='./outdata/zim/log_unet/')
    parser.add_argument('--model_path', type=str,
                        default='/public/home/zzuegr01/lhf/Experimeat_02/RSRNet/model/acc_81.h5')
                        #_1e-09_1.0_6.5_0.8_4_202109101846_model_at_epoch_9.h5')
    # ~/ljm/lpl_best_model/1e-09_1.0_5.0_202104060302_model_at_epoch_18.h5
    parser.add_argument('--ratio_ro', type=float, default='1')
    parser.add_argument('--ratio_bu', type=float, default='1')
    parser.add_argument('--ratio_bg', type=float, default='1')
    parser.add_argument('--ratio_no', type=float, default='1')
    parser.add_argument('--lam', type=float, default='0.4')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--optimizer_flag', type=int, default=1)
    

    args = parser.parse_args()
    return args

args = Create_args()
if args.noise!=0.0:
    args.label_db='/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/train/map_0_'+str(args.noise)+'_s'
print(args.label_db)
args.save_path =args.save_path  +'sdl/'+str(args.noise)+'/'
args.log_path = args.log_path +'sdl/'+str(args.noise)+'/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.log_path): os.makedirs(args.log_path)
# if not os.path.exists(args.image_save): os.makedirs(args.image_save)

acc_logger = lpl_logger.logger(args.log_path,
                               os.environ['SLURM_JOBID'] + '_' +str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro))


# 计算损失值
def generate_loss2(y_true, y_pred):
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T + 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor2(y_true[:, :, :, 0], y_true[:, :, :, 1])
    res = (1 - args.lam) * tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    return res

def generate_loss3(y_true, y_pred):
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T + 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor3(y_true[:, :, :, 0], y_true[:, :, :, 1], y_true[:, :, :, 2])
    res = (1 - args.lam) * tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    return res

def generate_loss4(y_true, y_pred):
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T + 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor4(y_true[:, :, :, 0], y_true[:, :, :, 1], y_true[:, :, :, 2], y_true[:, :, :, 3])
    res = (1 - args.lam) * tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    return res


def generate_loss5(y_true, y_pred):
    y0=1-y_true[:, :, :, args.T]
    res0 = -tf.reduce_sum(tf.multiply(y0, tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor5(y_true[:, :, :, 0], y_true[:, :, :, 1], y_true[:, :, :, 2], y_true[:, :, :, 3],y_true[:, :, :, 4])
    res = (1 - args.lam) * tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    return res


def generate_loss6(y_true, y_pred):
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, args.T + 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor6(y_true[:, :, :, 0], y_true[:, :, :, 1], y_true[:, :, :, 2], y_true[:, :, :, 3],y_true[:, :, :, 4], y_true[:, :, :, 5])
    res = (1 - args.lam) * tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    return res

# 基于概率队列tensor计算正则化项
def regu_tensor2(p1, p2):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    P = tf.concat([extend_p1, extend_p2], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    regu = -tf.reduce_sum(noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2))
    return regu

def regu_tensor3(p1, p2, p3):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    extend_p3 = tf.expand_dims(p3, 3)
    P = tf.concat([extend_p1, extend_p2, extend_p3], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    noisy3 = tf.where(varP > args.ratio_no * max_varP, p3, one)
    regu = -tf.reduce_sum(
        noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2) + noisy3 * tf.log(noisy3))
    return regu
def regu_tensor4(p1, p2, p3, p4):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    extend_p3 = tf.expand_dims(p3, 3)
    extend_p4 = tf.expand_dims(p4, 3)
    P = tf.concat([extend_p1, extend_p2, extend_p3, extend_p4], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    noisy3 = tf.where(varP > args.ratio_no * max_varP, p3, one)
    noisy4 = tf.where(varP > args.ratio_no * max_varP, p4, one)
    regu = -tf.reduce_sum(
        noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2) + noisy3 * tf.log(noisy3) + noisy4 * tf.log(noisy4))
    return regu
def regu_tensor5(p1, p2, p3, p4,p5):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    extend_p3 = tf.expand_dims(p3, 3)
    extend_p4 = tf.expand_dims(p4, 3)
    extend_p5 = tf.expand_dims(p5, 3)
    P = tf.concat([extend_p1, extend_p2, extend_p3, extend_p4, extend_p5], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    noisy3 = tf.where(varP > args.ratio_no * max_varP, p3, one)
    noisy4 = tf.where(varP > args.ratio_no * max_varP, p4, one)
    noisy5 = tf.where(varP > args.ratio_no * max_varP, p5, one)
    regu = -tf.reduce_sum(noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2) + noisy3 * tf.log(noisy3) + noisy4 * tf.log(noisy4) + noisy5 * tf.log(noisy5))
    return regu



def regu_tensor6(p1, p2, p3, p4, p5, p6):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    extend_p3 = tf.expand_dims(p3, 3)
    extend_p4 = tf.expand_dims(p4, 3)
    extend_p5 = tf.expand_dims(p5, 3)
    extend_p6 = tf.expand_dims(p6, 3)
    P = tf.concat([extend_p1, extend_p2, extend_p3, extend_p4, extend_p5, extend_p6], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    noisy3 = tf.where(varP > args.ratio_no * max_varP, p3, one)
    noisy4 = tf.where(varP > args.ratio_no * max_varP, p4, one)
    noisy5 = tf.where(varP > args.ratio_no * max_varP, p5, one)
    noisy6 = tf.where(varP > args.ratio_no * max_varP, p6, one)
    regu = -tf.reduce_sum(
        noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2) + noisy3 * tf.log(noisy3) + noisy4 * tf.log(noisy4)
        + noisy5 * tf.log(noisy5) + noisy6 * tf.log(noisy6))
    return regu


def regu_tensor8(p1, p2, p3, p4, p5, p6, p7, p8):
    # 提取目标（道路）的tensor
    extend_p1 = tf.expand_dims(p1, 3)
    extend_p2 = tf.expand_dims(p2, 3)
    extend_p3 = tf.expand_dims(p3, 3)
    extend_p4 = tf.expand_dims(p4, 3)
    extend_p5 = tf.expand_dims(p5, 3)
    extend_p6 = tf.expand_dims(p6, 3)
    extend_p7 = tf.expand_dims(p7, 3)
    extend_p8 = tf.expand_dims(p8, 3)

    P = tf.concat([extend_p1, extend_p2, extend_p3, extend_p4, extend_p5, extend_p6, extend_p7, extend_p8], 3)
    # 计算方差
    meanP, varP = tf.nn.moments(P, 3)
    max_varP = tf.reduce_max(varP)
    # 获取噪声索引
    one = tf.ones_like(p1)
    noisy1 = tf.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = tf.where(varP > args.ratio_no * max_varP, p2, one)
    noisy3 = tf.where(varP > args.ratio_no * max_varP, p3, one)
    noisy4 = tf.where(varP > args.ratio_no * max_varP, p4, one)
    noisy5 = tf.where(varP > args.ratio_no * max_varP, p5, one)
    noisy6 = tf.where(varP > args.ratio_no * max_varP, p6, one)
    noisy7 = tf.where(varP > args.ratio_no * max_varP, p7, one)
    noisy8 = tf.where(varP > args.ratio_no * max_varP, p8, one)
    regu = -tf.reduce_sum(
        noisy1 * tf.log(noisy1) + noisy2 * tf.log(noisy2) + noisy3 * tf.log(noisy3) + noisy4 * tf.log(noisy4) +
        noisy5 * tf.log(noisy5) + noisy6 * tf.log(noisy6) + noisy7 * tf.log(noisy7) + noisy8 * tf.log(noisy8))
    return regu


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, test_data, test_name_list_flat, period, model, labels):
        super(Metrics, self).__init__()
        self.test_data = test_data
        # self.test_label = test_label
        self.test_name_list_flat = test_name_list_flat
        self.period = period
        self.model_to_save = model
        self.labels = labels
    '''
    def on_train_begin(self, epoch, logs=None, path=args.image_save):
        global i
        print('printf acc...')
        a = time.time()
        probability_seg = multi_gpu_model.predict(test_data)
        # probability_seg = model.predict(test_data)
        pre, rec = acc.acc_2D_batch(probability_seg, self.labels)
        f1 = acc.acc_f1(pre, rec)
        print(epoch, pre, rec, f1)
        acc_logger.output([i, pre, rec, f1])
        self.model_to_save.save(
            args.save_path + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro) + '_'
            + str(args.ratio_no) + '_' + time.strftime("%Y%m%d%H%M",
                                                       time.localtime()) + '_' + 'model_at_epoch_%d.h5' % i)
        b = time.time()
        print("spend time :", b - a)
    '''
    def on_epoch_end(self, epoch, logs=None, path=args.image_save):
        # if  (epoch % self.period  == 0)&(epoch!=0)&(epoch!=1):
        global i
        print('printf acc...')
        pre_sum = 0
        rec_sum = 0
        f1_sum = 0

        a = time.time()

        probability_seg = multi_gpu_model.predict(test_data)
        pre, rec, iou = acc.acc_2D_batch(probability_seg, self.labels)
        #iou=' '
        b = time.time()

        print("spend time :", b - a)
        '''
        pre_labels, names = lpl_prepareData.merge_map_segmentation(probability_seg, test_name_list_flat)
        labels, names = lpl_prepareData.merge_map_segmentation(self.labels, test_name_list_flat)
        for k in range(0, len(pre_labels)):
            pre_label = pre_labels[k]
            label = labels[k]
            pre, rec = acc.acc_2D(pre_label, label)
            f1 = acc.acc_f1(pre, rec)
            # if path != None:
            # lpl_prepareData.save_image(label, path+time.strftime('%Y%m%d%H%M', time.localtime(time.time())) +'_'+ names[i] + '-'+str(epoch)+'_label.tif')
            # lpl_prepareData.save_image(pre_label, path +time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            # +'_'+str(args.lr)+'_'+str(args.ratio_bg)+'_'+str(args.ratio_ro)+'_'+ names[i] +'_'+str(epoch)+ '.tif')
            pre_sum = pre_sum + pre
            rec_sum = rec_sum + rec
        pre = pre_sum / len(pre_labels)
        rec = rec_sum / len(pre_labels)
        '''

        f1 = acc.acc_f1(pre, rec)
        print(epoch, pre, rec, f1, iou)

        acc_logger.output([i, pre, rec, f1, iou])
        #return
        self.model_to_save.save(
            args.save_path + os.environ['SLURM_JOBID'] + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro) + '_'
            + str(args.ratio_no) + '_'+str(args.T)+'_' + time.strftime("%Y%m%d%H%M",
                                                       time.localtime()) + '_' + 'model_at_epoch_%d.h5' % i)



# 加载数据，并将标签转化为标准标签'''

def load_data():
    batchs_data, batchs_label, key = lpl_prepareData.GetData_mul_gpu(args, 4, args.image_db,
                                                                     args.label_db)
    test_data, test_label, test_name_list = lpl_prepareData.GetData(args, args.test_image_db, args.test_label_db)
    a, b, train_name_list = lpl_prepareData.GetData(args, args.image_db, args.label_db)
    train_name_list_flat = []
    test_name_list_flat = []
    # 对影像快名字进行flat

    for i in range(len(test_name_list)):
        for j in range(len(test_name_list[i])):
            test_name_list_flat.append(test_name_list[i][j])
    for i in range(len(train_name_list)):
        for j in range(len(train_name_list[i])):
            train_name_list_flat.append(train_name_list[i][j])
    _train_data = []
    _train_label = []
    _test_label = []
    _test_data = []
    index = np.arange(len(batchs_data))
    for i in index:
        train_data = batchs_data[i]
        train_label = batchs_label[i]
        if len(train_data) == args.batchsize * 4:
            _train_data.extend(train_data)
            _train_label.extend(train_label)
    shape_data = _train_data[0].shape
    _train_data = np.reshape(_train_data, [-1, shape_data[0], shape_data[1], shape_data[2]])
    shape_label = _train_label[0].shape
    _train_label = np.reshape(_train_label, [-1, shape_label[0], shape_label[1], shape_label[2]])
    for i in range(0, len(test_data)):
        test_data_sub = test_data[i]
        _test_data.extend(test_data_sub)
    shape_data = _test_data[0].shape
    _test_data = np.reshape(_test_data, [-1, shape_data[0], shape_data[1], shape_data[2]])
    for i in range(0, len(test_label)):
        test_data_sub = test_label[i]
        _test_label.extend(test_data_sub)
    shape_label = _test_label[0].shape
    _test_label = np.reshape(_test_label, [-1, shape_label[0], shape_label[1], shape_label[2]])
    return _train_data, _train_label, _test_data, _test_label, test_name_list_flat, train_name_list_flat


def comput_squre_2(a, b, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2
    return sum_0
def comput_squre_3(a, b, c,n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2 + (n - c) ** 2
    return sum_0

def comput_squre_4(a, b, c, d, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2 + (n - c) ** 2 + (n - d) ** 2
    return sum_0


def comput_squre_5(a, b, c, d, e, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2 + (n - c) ** 2 + (n - d) ** 2 + (n - e) ** 2
    return sum_0


def comput_squre_6(a, b, c, d, e, f, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2 + (n - c) ** 2 + (n - d) ** 2 + (n - e) ** 2 + (n - f) ** 2
    return sum_0


def comput_squre_8(a, b, c, d, e, f, g, h, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2 + (n - c) ** 2 + (n - d) ** 2 + (n - e) ** 2 + (n - f) ** 2 + (n - g) ** 2 + (
                n - h) ** 2
    return sum_0


# 输入参数:p1,p2,p3,p4,y

def comput_entropy_2(p1, p2, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 2))
    for i in range(count):
        # 抽取道路通道
        #print("----")

        #print(p1.shape)
        #print(pi[:, :, 0].shape)
        pi[:, :, 0] = p1[i,:, :,0]
        pi[:, :, 1] = p2[i,:, :,0]
        #print(pi[:, :, 0].shape)
        #print("----")

        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_2(pi[:, :, 0], pi[:, :, 1], 1)
        sq_0 = comput_squre_2(pi[:, :, 0], pi[:, :, 1], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1];
        y1[noise_index] = y_noise_corrected;
        y0 = 1 - y1
        yc[:, :, 1] = y1;
        yc[:, :, 0] = y0;
        y_update.append(yc)
    return y_update

def comput_entropy_3(p1, p2, p3, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 3))

    for i in range(count):
        # 抽取道路通道
        pi[:, :, 0] = p1[i, :, :, 0]
        pi[:, :, 1] = p2[i, :, :, 0]
        pi[:, :, 2] = p3[i, :, :, 0]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_3(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], 1)
        sq_0 = comput_squre_3(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1];
        y1[noise_index] = y_noise_corrected;
        y0 = 1 - y1
        yc[:, :, 1] = y1;
        yc[:, :, 0] = y0;
        y_update.append(yc)
    return y_update

def comput_entropy_4(p1, p2, p3, p4, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 4))

    for i in range(count):
        # 抽取道路通道
        pi[:, :, 0] = p1[i, :, :, 0]
        pi[:, :, 1] = p2[i, :, :, 0]
        pi[:, :, 2] = p3[i, :, :, 0]
        pi[:, :, 3] = p4[i, :, :, 0]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_4(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], 1)
        sq_0 = comput_squre_4(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1];
        y1[noise_index] = y_noise_corrected;
        y0 = 1 - y1
        yc[:, :, 1] = y1;
        yc[:, :, 0] = y0;
        y_update.append(yc)
    return y_update


def comput_entropy_5(p1, p2, p3, p4, p5, y):
    print("begin")
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 5))
    for i in range(count):
        # 抽取道路通道
        pi[:, :, 0] = p1[i, :, :, 0]
        pi[:, :, 1] = p2[i, :, :, 0]
        pi[:, :, 2] = p3[i, :, :, 0]
        pi[:, :, 3] = p4[i, :, :, 0]
        pi[:, :, 4] = p5[i, :, :, 0]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_5(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], 1)
        sq_0 = comput_squre_5(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :] #128 128 1
        y1 = yc[:, :, 0]; #128 128
        y1[noise_index] = y_noise_corrected;
        #y0 = 1 - y1
        yc[:, :, 0] = y1;
        #print("yc.shape:",yc.shape)
        #yc[:, :, 0] = y0;
        y_update.append(yc)
    print("begin")
    return y_update


def comput_entropy_6(p1, p2, p3, p4, p5, p6, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 6))
    for i in range(count):
        # 抽取道路通道
        pi[:, :, 0] = p1[i, :, :, 0]
        pi[:, :, 1] = p2[i, :, :, 0]
        pi[:, :, 2] = p3[i, :, :, 0]
        pi[:, :, 3] = p4[i, :, :, 0]
        pi[:, :, 4] = p5[i, :, :, 0]
        pi[:, :, 5] = p6[i, :, :, 0]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_6(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], pi[:, :, 5], 1)
        sq_0 = comput_squre_6(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], pi[:, :, 5], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1];
        y1[noise_index] = y_noise_corrected;
        y0 = 1 - y1
        yc[:, :, 1] = y1;
        yc[:, :, 0] = y0;
        y_update.append(yc)
    return y_update


def comput_entropy_8(p1, p2, p3, p4, p5, p6, p7, p8, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 8))
    for i in range(count):
        # 抽取道路通道10+15+60+15+60+30+30+若干
        pi[:, :, 0] = p1[i, :, :, 1]
        pi[:, :, 1] = p2[i, :, :, 1]
        pi[:, :, 2] = p3[i, :, :, 1]
        pi[:, :, 3] = p4[i, :, :, 1]
        pi[:, :, 4] = p5[i, :, :, 1]
        pi[:, :, 5] = p6[i, :, :, 1]
        pi[:, :, 6] = p7[i, :, :, 1]
        pi[:, :, 7] = p8[i, :, :, 1]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # print(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_8(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], pi[:, :, 5], pi[:, :, 6],
                              pi[:, :, 7], 1)
        sq_0 = comput_squre_8(pi[:, :, 0], pi[:, :, 1], pi[:, :, 2], pi[:, :, 3], pi[:, :, 4], pi[:, :, 5], pi[:, :, 6],
                              pi[:, :, 7], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1];
        y1[noise_index] = y_noise_corrected;
        y0 = 1 - y1
        yc[:, :, 1] = y1;
        yc[:, :, 0] = y0;
        y_update.append(yc)
    return y_update


if __name__ == '__main__':
    t_label = 0

    train_data, train_label, test_data, test_label, test_name_list_flat, train_name_list_flat = load_data()

    # a = np.array(train_data)
    with tf.device("/gpu:0"):
        if os.path.exists(args.model_path):
            model = NN.resnet(128)
            model.load_weights(args.model_path,  by_name=True)
            #model = models.load_model(args.model_path, compile=False)

            #model.summary()
            print("checkpoint_loaded success!")
        else:
            model = NN.resnet(128)
            #model = t_unet.build(args.sat_size)
            #model.summary()
    #multi_gpu_model = multi_gpu_model(model, gpus=[0, 1,2,3])
    multi_gpu_model = model
    #multi_gpu_model.summary()
    learning_rate = args.lr
    decay_rate = learning_rate / args.decay_steps
    momentum = 0.9
    
    if args.optimizer_flag==1:
        optimizer_ = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    elif args.optimizer_flag==2:
        optimizer_ = adam(lr=learning_rate, decay=decay_rate)
    elif args.optimizer_flag==3:
        optimizer_ = adadelta(lr=learning_rate, decay=decay_rate)
    else:
        optimizer_ = adagrad(lr=learning_rate, decay=decay_rate)

    #sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    if args.T==2:
        multi_gpu_model.compile(loss=generate_loss2,
                                optimizer=optimizer_
                                )
    elif args.T==3:
        multi_gpu_model.compile(loss=generate_loss3,
                                optimizer=optimizer_
                                )
    elif args.T==4:
        multi_gpu_model.compile(loss=generate_loss4,
                                optimizer=optimizer_
                                )
    elif args.T==6:
        multi_gpu_model.compile(loss=generate_loss6,
                                optimizer=optimizer_
                                )
    else:
        multi_gpu_model.compile(loss=generate_loss5,
                                optimizer=optimizer_
                                )
    for x in model.inputs:
        print("-----------", x.device)
    # labels, names = lpl_prepareData.merge_map_segmentation(test_label, test_name_list_flat)
    callbacks = [
        Metrics(period=args.period, test_data=test_data, test_name_list_flat=test_name_list_flat, model=model,
                labels=test_label)
        # keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, mode='accuracy', save_best_only=False，
        # verbose=1, save_weights_only=False,period=3)
    ]

    class_weights = np.zeros((16, 16, 2))
    class_weights[:, :, :0] += 1
    class_weights[:, :, :1] += 1
    # 训练模型
    epoch = 200
    step = args.T
    # 初始化当前标签
    current_labels = train_label
    sdl = []  # 标签概率队列
    # #初始化标签队列
    # for i in range (step):
    #     sdl.append(y_correct)
    # 初始化多重训练标签 10+20加载数据+15初始+60+15g
    probability_seg = multi_gpu_model.predict(train_data)
    #print("probability_seg",probability_seg.shape)
    # probability_seg=np.expand_dims(probability_seg,-1)
    mul_labels = probability_seg[:,:,:,1,None]
    #mul_labels=mul_labels[None,:,:,:]
    #print("probability_seg[:,:,:,1]", probability_seg[:,:,:,1,None].shape)

    #15+30
    if(args.T==5):
        for i in range(args.T - 1):
            mul_labels = np.concatenate((mul_labels, probability_seg[:,:,:,1,None]), -1)
        mul_labels = np.concatenate((mul_labels, train_label[:,:,:,1,None]), -1)
    else:
        for i in range(args.T - 1):
            mul_labels = np.concatenate((mul_labels, probability_seg[:,:,:,1,None]), -1)
        mul_labels = np.concatenate((mul_labels, train_label[:, :, :, 0, None]), -1)
        mul_labels = np.concatenate((mul_labels, train_label[:,:,:,1,None]), -1)
    #print("mul_labels.shape：",mul_labels.shape)
    
    probability_seg_= multi_gpu_model.predict(test_data)
        # probability_seg = model.predict(test_data)
    pre, rec = acc.acc_2D(probability_seg_, test_label)
    f1 = acc.acc_f1(pre, rec)
    print(args.model_path)
    
    print("test", pre, rec, f1)
    acc_logger.output([args.model_path])
    acc_logger.output(["test", pre, rec, f1])
    # exit()
    for i in range(epoch):
        print("This is no.%d " % (i))

        multi_gpu_model.fit(x=train_data, y=mul_labels, callbacks=callbacks, epochs=1,
                            batch_size=args.batchsize, verbose=1, shuffle=True, sample_weight=None)
        probability_seg = multi_gpu_model.predict(train_data)
        if args.T == 2:
            if len(sdl) <= args.T - 1:
                sdl.append(probability_seg[:,:,:,1,None])
                if len(sdl)==2:
                  corrrect_label = comput_entropy_2(sdl[0], sdl[1], train_label)
                  current_labels = np.array(corrrect_label)
                  sdl.append(current_labels[:, :, :, 0,None])
                  sdl.append(current_labels[:,:,:,1,None])
                  # 更新多重训练标签
                  mul_labels = np.array(sdl[0]);
                  for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                  sdl.pop()
                  sdl.pop()
                  
            else:
                # 删除最早的预测标签概率
                sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:,:,:,1,None])
                # 纠正当前标签
                corrrect_label = comput_entropy_2(sdl[0], sdl[1], train_label)
                current_labels = np.array(corrrect_label)
                sdl.append(current_labels[:, :, :, 0,None])
                sdl.append(current_labels[:,:,:,1,None])
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                sdl.pop()
                sdl.pop()
        elif args.T == 3:
            if len(sdl) <= args.T - 1:
                sdl.append(probability_seg[:, :, :, 1, None])
            else:
                # 删除最早的预测标签概率
                sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:, :, :, 1, None])
                # 纠正当前标签
                corrrect_label = comput_entropy_3(sdl[0], sdl[1], sdl[2],train_label)
                current_labels = np.array(corrrect_label)
                sdl.append(current_labels[:, :, :, 0, None])
                sdl.append(current_labels[:, :, :, 1, None])
                #sdl.append(current_labels)
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                sdl.pop()
                sdl.pop()
        elif args.T == 4:
            if len(sdl) <= args.T - 1:
                sdl.append(probability_seg[:, :, :, 1, None])
            else:
                # 删除最早的预测标签概率
                sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:, :, :, 1, None])
                # 纠正当前标签
                corrrect_label = comput_entropy_4(sdl[0], sdl[1], sdl[2], sdl[3], train_label)
                current_labels = np.array(corrrect_label)
                sdl.append(current_labels[:, :, :, 0, None])
                sdl.append(current_labels[:, :, :, 1, None])
                #sdl.append(current_labels)
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                sdl.pop()
                sdl.pop()
        elif args.T == 6:
            if len(sdl) <= args.T - 1:
                sdl.append(probability_seg[:, :, :, 1, None])
            else:
                # 删除最早的预测标签概率
                sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:, :, :, 1, None])
                # 纠正当前标签
                corrrect_label = comput_entropy_6(sdl[0], sdl[1], sdl[2], sdl[3],sdl[4], sdl[5],train_label)
                current_labels = np.array(corrrect_label)
                sdl.append(current_labels[:, :, :, 0, None])
                sdl.append(current_labels[:, :, :, 1, None])
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                sdl.pop()
                sdl.pop()
        elif args.T == 8:
            if len(sdl) <= args.T - 1:
                sdl.append(probability_seg[:, :, :, 1, None])
            else:
                # 删除最早的预测标签概率
                sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:, :, :, 1, None])
                # 纠正当前标签
                corrrect_label = comput_entropy_8(sdl[0], sdl[1], sdl[2], sdl[3], sdl[4], sdl[5], sdl[6], sdl[7],
                                                  train_label)
                current_labels = np.array(corrrect_label)
                sdl.append(current_labels[:, :, :, 0, None])
                sdl.append(current_labels[:, :, :, 1, None])
                #sdl.append(current_labels)
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                sdl.pop()
                sdl.pop()
        else:
            if len(sdl)!=args.T-1:
                sdl.append(probability_seg[:, :, :, 1, None])
            else:
                # 删除最早的预测标签概率
                #sdl.pop(0);
                # 添加当前预测标签概率
                sdl.append(probability_seg[:, :, :, 1, None])
                # 纠正当前标签
                corrrect_label = comput_entropy_5(sdl[0], sdl[1], sdl[2], sdl[3], sdl[4],train_label[:, :, :, 1, None])
                current_labels = np.array(corrrect_label)
                #sdl.append(current_labels[:, :, :, 0, None])
                sdl.append(current_labels)
                #sdl.append(current_labels)
                # 更新多重训练标签
                mul_labels = np.array(sdl[0]);
                for j in range(1, len(sdl)):
                    item = np.array(sdl[j])
                    mul_labels = np.concatenate((mul_labels, item), -1)
                #sdl.pop()
                sdl.pop()
                sdl.pop(0)
    st = time.time()
    print('epochs:{}'.format(time.time() - st))
    # 模型评估

    time1 = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(time1)
