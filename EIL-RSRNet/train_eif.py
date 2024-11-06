# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import socket
import numpy as np
import time
import argparse
from matplotlib import pyplot as plt

# 降低tensorflow中log的等级，减少警告和提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"

import tensorflow as tf
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import horovod.tensorflow.keras as hvd
from mpi4py import MPI

sys.path.append("./standard")

# import lpl_prepareData_1C as lpl_prepareData
import lpl_accuracy as acc
import lpl_prepareData
import lpl_logger
import lpl_tensorboard
import lpl_accuracy as acc
import random
# 导入模型包
# from CBR_model_test import build
from rernet import resnet
import cv2 as cv
# 精确度计算
sys.path.append("/public/home/zzuegr01/lhf/MUL/CBRNet-main_ex")
import lpl_accuracy1 as lpl_accuracy

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# horovod初始化
hvd.init()

# 不进行如下设置会报内存错误
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.compat.v1.Session(config=config))

str_rank = str(hvd.rank())


# 定义参数
def Create_args():
    parser = argparse.ArgumentParser()
    # 常用
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    # 是否保存
    parser.add_argument('--save_picture_or_not', type=bool, default=False)
    parser.add_argument('--save_model_or_not', type=bool, default=True)
    # 图片参数
    parser.add_argument('--wro_weight', type=int, default=4)
    parser.add_argument('--sat_size', type=int, default=128)
    parser.add_argument('--sat_channel', type=int, default=3)
    parser.add_argument('--map_size', type=int, default=128)
    parser.add_argument('--map_channel', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    # gpu设置
    parser.add_argument('--gpu_index', type=str, default="0")
    parser.add_argument('--gpu_rate', type=float, default=0.5)
    # 优化器参数
    parser.add_argument('--lr', type=float, default=1e-7)  # 设置学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)  # 设置衰减率
    parser.add_argument('--decay_steps', type=float, default=100000)
    parser.add_argument('--momentum', type=float, default=0.9)
    # 训练集路径
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
    # 结果保存路径
    
    parser.add_argument('--image_save', type=str, default='./log/image/0/process_' + str_rank + '/')
    parser.add_argument('--save_path', type=str, default='./log/save_model/0/process_' + str_rank + '/')
    parser.add_argument('--log_path', type=str, default='./log/log/0/process_' + str_rank + '/')
    parser.add_argument('--model_path', type=str, default='./model/efi/0/' + str_rank + '/')
    '''
    parser.add_argument('--image_save', type=str, default='./log/image/process_' + str_rank + '/')
    parser.add_argument('--save_path', type=str, default='./log/save_model/process_' + str_rank + '/')
    parser.add_argument('--log_path', type=str, default='./log/log/process_' + str_rank + '/')
    parser.add_argument('--model_path', type=str, default='./model_list/model_' + str_rank + '/')
    '''
    
    
    # 损失相关参数
    parser.add_argument('--ratio_ro', type=float, default=1)
    parser.add_argument('--ratio_bu', type=float, default=1)
    parser.add_argument('--ratio_bg', type=float, default=1)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--a0', type=float, default=0.1)
    parser.add_argument('--a1', type=float, default=2)
    # horovod参数
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--comm', default=None)
    parser.add_argument('--size', type=int, default=None)
    args = parser.parse_args()
    return args


args = Create_args()

args.comm = comm
args.rank = rank
args.size = size


# 若文件夹不存在则创建，在多进程中不使用try容易报错
def make_file(file):
    if not os.path.exists(file):
        try:
            os.makedirs(file)
        except OSError:
            pass


make_file(args.save_path)
make_file(args.log_path)
make_file(args.image_save)
make_file(args.model_path)

# 设置保存路径和文件名
common_name = os.environ['SLURM_JOBID'] + '_' + str(args.lr) + '_' + str(args.batchsize) + '_' + str(
    args.a0) + '_' + str(args.a1) + '_' + str(args.lam) + '_' + str(args.ratio_ro)
acc_logger = lpl_logger.logger(args.log_path,
                               common_name)


# 计算损失值，加1e-5防止log值过小为负无穷
def generate_loss(y_true, y_pred):
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 0], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    res2 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 2], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res3 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 3], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    res = tf.reduce_sum(args.ratio_bg * res0 + args.ratio_ro * res1)
    res0 = tf.reduce_sum(args.ratio_bg * res2 + args.ratio_ro * res3)
    return res + args.lam * res0
    # return res


class Metrics(tf.keras.callbacks.Callback):

    def __init__(self, test_name_list_flat, period, model):
        super(Metrics, self).__init__()
        self.epoch_times = []
        self.compute_times = []
        self.test_name_list_flat = test_name_list_flat
        self.period = period
        self.model_to_save = model

    '''
    def on_train_begin(self,logs=None, path=args.image_save):

            print('printf acc...')
            pre_sum = 0
            rec_sum = 0
            f1_sum = 0
            a = time.time()

            compute_time = time.time()
            pre_sum = 0
            rec_sum = 0
            probability_seg = model.predict(train_data)
        # 设置数组存放所有概率
            all_seg = np.zeros((hvd.size(),) + train_data.shape, dtype='f')
        # 将所有进程的probability_seg放入all_seg中
            comm.Allgather([probability_seg, MPI.FLOAT], [all_seg, MPI.FLOAT])
            prop = np.zeros([4,all_seg.shape[1],128,128])
            #从四个不同的模型中抽取概率
            prop[0]=all_seg[0,:,:,:,1]
            prop[1]=all_seg[1,:,:,:,1]
            prop[2]=all_seg[2,:,:,:,1]
            prop[3]=all_seg[3,:,:,:,1]
            for i in range (prop.shape[1]):
                a0 = 0.1
                a1 = 2
                #pro = p0 = p1 = 0
                var = np.zeros([5, 128, 128])
                #vartest = np.zeros([4, 128, 128])
                var[0]=prop[0,i,:,:]
                var[1]=prop[1,i,:,:]
                var[2]=prop[2,i,:,:]
                var[3]=prop[3,i,:,:]
                var[4]=self.train_labels[i,:,:,1]
                for j in range(5):
                    pro = var[:, :, j]
                    p0 = 1 - pro
                    p1 = pro
                    r0 = 1 / (1 + np.exp(-a0 * p0))
                    r1 = 1 / (1 + np.exp(-a1 * p1))
                    c1 = np.where(p1 >= 0.5, 1, 0)
                    c0 = 1 - c1
                    a1 = a1 - a1 * r1 * (c1 - r1)
                    a0 = a0 - a0 * r0 * (c0 - r0)
                rc0 = 0;
                rc1 = 0
                for j in range(5):
                    pro = var[:, :, j]
                    p0 = 1 - pro
                    p1 = pro
                    r0 = 1 / (1 + np.exp(-a0 * p0))
                    r1 = 1 / (1 + np.exp(-a1 * p1))
                    rc0 = rc0 + r0 * p0
                    rc1 = rc1 + r1 * p1
                    rc0 = rc0 + r0 * p0
                    rc1 = rc1 + r1 * p1

                index_bulding = np.where(rc1 >= rc0)
                y = np.zeros([128, 128])
                y[index_bulding] = 1
                x=1-y

                #newlabels[i,:,:,0]=self.train_labels[i,:,:,0]
                #newlabels[i,:,:,1]=self.train_labels[i,:,:,1]
                newlabels[i,:,:,2]=x
                newlabels[i,:,:,3]=y
            b = time.time()
            print("spend time :", b - a)
        # pre_labels, names = lpl_prepareData.merge_map_segmentation(probability_seg, test_name_list_flat)
        #
        # for i in range(0, len(pre_labels)):
        #     pre_label = pre_labels[i]
        #     label = labels[i]
        #     pre, rec = acc.acc_2D(pre_label, label)
        #     f1 = acc.acc_f1(pre, rec)
        #     if path != None:
        #         lpl_prepareData.save_image(label,
        #                                    path + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '_' +
        #                                    names[i] + '-' + str(epoch) + '_label.tif')
        #         lpl_prepareData.save_image(pre_label,
        #                                    path + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '_' +
        #                                    names[i] + '-' + str(epoch) + '.tif')
        #     pre_sum = pre_sum + pre
        #     rec_sum = rec_sum + rec
        #     f1 = acc.acc_f1(pre, rec)
        #     print(epoch, pre, rec, f1)
        #
        # pre = pre_sum / len(pre_labels)
        # rec = rec_sum / len(pre_labels)
        # f1 = acc.acc_f1(pre, rec)
        # print(epoch, pre, rec, f1)
        #
        # acc_logger.output([epoch, pre, rec, f1])
        # exit()
    '''

    def on_epoch_begin(self, epoch, logs=None):
        self.probability_test_seg = np.zeros([len(test_data), 128, 128, 2], dtype=np.float16)


        t3 = time.time()
        probability_seg = model.predict(test_data)
        probability_seg = probability_seg.astype(np.float16)

        t4 = time.time()
        print('test_data预测获取成功!', t4 - t3)

        dim0 = probability_seg.shape[0]
        # 存放所有概率
        all_seg = np.zeros((size, dim0, 128, 128, 2), dtype=np.float16)
        # ldim为一次传递的数量，不要大于65536
        ldim = 30000
        i = 0
        while True:
            if (i + 1) * ldim > dim0:
                tem0 = np.zeros((size,) + (dim0 - i * ldim, 128, 128, 2), dtype=np.float16)
                tem = np.array(probability_seg[i * ldim:dim0])
                comm.Allgather([tem, MPI.FLOAT], [tem0, MPI.FLOAT])
                all_seg[:, i * ldim:dim0] = tem0
                break
            else:
                tem0 = np.zeros((size,) + (ldim, 128, 128, 2), dtype=np.float16)
                tem = np.array(probability_seg[i * ldim:(i + 1) * ldim])
                comm.Allgather([tem, MPI.FLOAT], [tem0, MPI.FLOAT])
                all_seg[:, i * ldim:(i + 1) * ldim] = tem0
            i += 1

        t5 = time.time()
        print('test_data传递成功!', t5 - t4)

        # proptest = np.zeros([4, all_seg.shape[1], 128, 128], np.float16)
        # # 从四个不同的模型中抽取概率
        #
        # proptest[0] = all_seg[0, :, :, :, 1]
        # proptest[1] = all_seg[1, :, :, :, 1]
        # proptest[2] = all_seg[2, :, :, :, 1]
        # proptest[3] = all_seg[3, :, :, :, 1]
        # # proptest3 = all_seg[3, :, :, :, 1]

        # test_0 = all_seg[0, :, :, :, :]
        # vartest_list = np.zeros([all_seg.shape[1],128, 128,2])

        for i in range(len(test_data)):
            a0 = 0.1
            a1 = 2
            # pro = p0 = p1 = 0

            var = np.zeros([size, 128, 128], dtype=np.float16)
            # vartest = np.zeros([4, 128, 128])
            for index in range(size):
              var[index] = all_seg[index, i, :, :, 1]

            '''
            var0 = proptest0[i]
            var1 = proptest1[i]
            var2 = proptest2[i]
            var3 = proptest3[i]
            var0 = var0[:,:,1]
            var1 = var1[:,:,1]
            var2 = var2[:,:,1]
            var3 = var3[:,:,1]
            '''
            # var3 = proptest3[3, i, :, :]
            '''

            '''
            for j in range(size):
                pro = var[j, :, :]
                p0 = 1 - pro
                p1 = pro
                r0 = 1 / (1 + np.exp(-a0 * p0))
                r1 = 1 / (1 + np.exp(-a1 * p1))
                c1 = np.where(p1 >= 0.5, 1, 0)
                c0 = 1 - c1
                a1 = a1 - a1 * r1 * (c1 - r1)
                a0 = a0 - a0 * r0 * (c0 - r0)
            rc0 = 0
            rc1 = 0
            for j in range(size):
                pro = var[j, :, :]
                p0 = 1 - pro
                p1 = pro
                r0 = 1 / (1 + np.exp(-a0 * p0))
                r1 = 1 / (1 + np.exp(-a1 * p1))
                rc0 = rc0 + r0 * p0
                rc1 = rc1 + r1 * p1

            index_bulding = np.where(rc1 >= rc0)
            y = np.zeros([128, 128])
            y[index_bulding] = 1
            x = 1 - y
            # 构建测试集
            self.probability_test_seg[i, :, :, 0] = x
            self.probability_test_seg[i, :, :, 1] = y
        # print(self.probability_test_seg)
        t6 = time.time()
        print('test_data计算完成!', t6 - t5)

        compute_time = time.time()
        pre_sum = 0
        rec_sum = 0
        pre_labels, names = lpl_prepareData.merge_map_segmentation(self.probability_test_seg, test_name_list_flat)
        avg_TP = 0;avg_FP = 0;avg_FN = 0;
        for i in range(0, len(pre_labels)):
            pre_label = pre_labels[i]
            label = labels[i]
            # 单张精确度
            TP, FP, FN = lpl_accuracy.acc_2D(pre_label, label)
            avg_TP = avg_TP + TP
            avg_FP = avg_FP + FP
            avg_FN = avg_FN + FN
            # if path != None:
            # lpl_prepareData.save_image(label, path+time.strftime('%Y%m%d%H%M', time.localtime(time.time())) +'_'+ names[i] + '-'+str(epoch)+'_label.tif')
            # lpl_prepareData.save_image(pre_label, path + time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            #                            + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(
            #     args.ratio_ro) + '_' + names[i].strip('.tif') + '_' + str(epoch) + '.tif')
        precision = float(avg_TP) / float(avg_TP + avg_FP + 0.0001)
        recall = float(avg_TP) / float(avg_TP + avg_FN + 0.0001)
        ave_f1 = lpl_accuracy.acc_f1(precision, recall)
        iou = float(avg_TP) / float(avg_TP + avg_FP + avg_FN + 0.0001)
        print(epoch, precision, recall, ave_f1, iou)

        t7 = time.time()
        print('精确度计算完成!', t7 - t6)
        # 强制同步
        signal = 0
        if not rank == 0:
            # recv会阻塞接收者进程
            comm.recv(source=rank - 1)
            time.sleep(0.01)
        print('hvd.rank', hvd.rank(), ':', epoch, precision, recall, ave_f1, iou)
        if not rank == size - 1:
            # send不会阻塞发起者进程
            comm.send(signal, dest=rank + 1)
        # 所有进程进行同步
        comm.bcast(signal, root=size - 1)

        acc_logger.output(
            [epoch, precision, recall, ave_f1, iou])

        # 是否保存模型
        if args.save_model_or_not and args.save_path is not None:
            make_file(args.save_path + common_name + '/')
            model.save_weights(
                args.save_path + common_name + '/' + common_name + '_' + time.strftime('%Y%m%d%H%M', time.localtime(
                    time.time())) + '_' + str(args.lr) + '_' + str(
                    args.ratio_ro) + '_' + 'model_at_epoch_%d.h5' % epoch)
            self.compute_times.append(time.time() - compute_time)
        print('顺利完成!')
        # exit()


# 加载数据，并将标签转化为标准标签'''
def load_data():
    batchs_data, batchs_label, key = lpl_prepareData.GetData_mul_gpu(args, 4, args.image_db,
                                                                     args.label_db)
    test_data, test_label, test_name_list = lpl_prepareData.GetData(args, args.test_image_db, args.test_label_db)
    test_name_list_flat = []
    # 对影像快名字进行flat
    for i in range(len(test_name_list)):
        for j in range(len(test_name_list[i])):
            test_name_list_flat.append(test_name_list[i][j])
    _train_data = []
    _train_label = []
    _test_label = []
    _test_data = []
    index = np.arange(len(batchs_data))
    print('len(batchs_data)=', len(batchs_data))
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
    return _train_data, _train_label, _test_data, _test_label, test_name_list_flat


# 主函数开始
train_data, train_label, test_data, test_label, test_name_list_flat = load_data()
model = resnet(128)
model_list = [os.path.join(args.model_path, name) for name in os.listdir(args.model_path)]
newlabels = np.zeros([len(train_label), 128, 128, 4], dtype='f')
if model_list:
    latest_model = max(model_list, key=os.path.getctime)
    model.load_weights(latest_model)
    print('进程号:' + str_rank, '---------------------成功加载模型参数:', latest_model, '---------------------------')
else:
    print('进程号:' + str_rank, '---------------------未加载模型!------------------------')
learning_rate = args.lr
decay_rate = learning_rate / args.decay_steps
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
model.compile(loss=generate_loss,
              optimizer=sgd,
              )

labels, names = lpl_prepareData.merge_map_segmentation(test_label, test_name_list_flat)
M_callback = Metrics(period=args.period, test_name_list_flat=test_name_list_flat, model=model)

callbacks = [M_callback, ]

# 设置权重
class_weights = np.zeros((16, 16, 2))
class_weights[:, :, :0] += 1
class_weights[:, :, :1] += 1
newlabels = np.concatenate((train_label, train_label), -1)

history = model.fit(x=train_data, y=newlabels, epochs=args.epochs, callbacks=callbacks,
                    batch_size=args.batchsize, verbose=1 if hvd.rank() == 0 else 0)

# history.history为字典，一般为{'loss': [597963.2665, 479606.214125]}
# np.save('./time_epoch/epoch_times', M_callback.epoch_times)
# np.save('./time_compute/compute_times', M_callback.compute_times)
# np.save('./loss_save/my_loss.npy', history.history['loss'])
