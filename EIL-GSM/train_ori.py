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
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
# import lpl_prepareData_1C as lpl_prepareData
import lpl_accuracy as acc
import lpl_prepareData
import lpl_logger
import lpl_tensorboard
import lpl_accuracy as acc
import socket
from model import gsm as NN

from keras import losses
from keras import optimizers
from keras.utils import multi_gpu_model
from keras import models
# from segnet_quick import build


from keras.utils import plot_model
from keras.optimizers import SGD


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
    parser.add_argument('--lr', type=float, default=1e-8)  # 设置学习率
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
                        default='')
                        #_1e-09_1.0_6.5_0.8_4_202109101846_model_at_epoch_9.h5')
    # ~/ljm/lpl_best_model/1e-09_1.0_5.0_202104060302_model_at_epoch_18.h5
    parser.add_argument('--ratio_ro', type=float, default='1')
    parser.add_argument('--ratio_bu', type=float, default='1')
    parser.add_argument('--ratio_bg', type=float, default='1')
    parser.add_argument('--ratio_no', type=float, default='0.8')
    parser.add_argument('--lam', type=float, default='0.4')
    parser.add_argument('--T', type=int, default=2)
    args = parser.parse_args()
    return args

args = Create_args()
if args.noise!=0.0:
    args.label_db='/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/train/map_0_'+str(args.noise)+'_s'
print(args.label_db)
args.save_path =args.save_path  +'oncs/'+str(args.noise)+'/'
args.log_path = args.log_path +'oncs/'+str(args.noise)+'/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.log_path): os.makedirs(args.log_path)
# if not os.path.exists(args.image_save): os.makedirs(args.image_save)

acc_logger = lpl_logger.logger(args.log_path,
                               str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro))


# 计算损失值
def generate_loss(y_true, y_pred):

    #if(len([y_true])!=len([y_pred])):
    print("2-4")

    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 0], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    #res2 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 2], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    #res3 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 3], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    #res = tf.reduce_sum(args.ratio_bg*res0 + args.ratio_ro*res1 + 0.2 * (args.ratio_bg*res2 + args.ratio_ro*res3))
    res = tf.reduce_sum(args.ratio_bg*res0 + args.ratio_ro*res1)

    #res = tf.reduce_sum(res0 + args.ratio_ro * res1)
    '''
    else:

    print("2 loss")
    print(len([y_true]))
    print(len([y_pred]))
    print("2-2")
    res0 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 0], tf.log(tf.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -tf.reduce_sum(tf.multiply(y_true[:, :, :, 1], tf.log(tf.add(1e-7, y_pred[:, :, :, 1]))))
    #a = np.array(res0+res1)
    #print("np.mean(a)",tf.reduce_mean(res0+res1))
    res = tf.reduce_sum(res0+1.5*res1)
    '''
    return res

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
        #probability_seg = model.predict(test_data)
        pre, rec = acc.acc_2D_batch(probability_seg, self.labels)
        f1 = acc.acc_f1(pre, rec)
        print(epoch, pre, rec, f1)
        acc_logger.output([i, pre, rec, f1])
        self.model_to_save.save(args.save_path + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro)+ '_'
                                +str(args.ratio_no)+'_'+time.strftime("%Y%m%d%H%M", time.localtime())+'_'+'model_at_epoch_%d.h5' % i)
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
        pre, rec,iou = acc.acc_2D_batch(probability_seg, self.labels)
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
            args.save_path + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro) + '_'
            + str(args.ratio_no) + '_'+str(args.T)+'_' + time.strftime("%Y%m%d%H%M",
                                                       time.localtime()) + '_' + 'model_at_epoch_%d.h5' % i)

# 加载数据，并将标签转化为标准标签'''

def load_data():
    batchs_data, batchs_label, key = lpl_prepareData.GetData_mul_gpu(args, 4, args.image_db,
                                                                     args.label_db)
    test_data, test_label, test_name_list = lpl_prepareData.GetData(args, args.test_image_db, args.test_label_db)
    a,b,train_name_list = lpl_prepareData.GetData(args, args.image_db, args.label_db)
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
    return _train_data, _train_label, _test_data, _test_label, test_name_list_flat,train_name_list_flat


def comput_squre_2(a,n):
     return (n-a)**2
def comput_squre_1(a, b, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2
    return sum_0
#输入参数:p1,p2,p3,p4,y
def comput_entropy_1(probility,y):
    K = 0.5
    y_update = []
    count = len(probility)
    shape=probility.shape
    pi=np.zeros((shape[1],shape[2],2))

    for i in range(count):
        #分别抽取背景和道路概率
        pi[:,:,0]=probility[i,:, :, 0]
        pi[:, :, 1] = probility[i, :, :, 1]
        #将道路概率转为矩阵
        road_mutex = np.where(pi[:, :, 1]>0.5,1,0)

        #uncertainty map U composed of the uncertainty values of all pixels on an image
        #将道路和背景概率对数相乘，获取每个像素的不确定值矩阵
        res0 = pi[:, :, 0]*np.log(pi[:, :, 0])
        res1 = pi[:, :, 1]*np.log(pi[:, :, 1])
        u = -(res0+res1)

        # 获取不确定值超过u平均值的像素即噪声像素的索引位置

        mean_u = np.mean(u) #求得不确定平均值

        if(mean_u<=K):
            vi = K
        else:
            vi = mean_u
        uncertain_index = np.where(u >= vi)
        #true_index = np.where(u < vi)
        #开始纠正标签
        #sq_1 = comput_squre_1(pi[:,:,0],pi[:,:,1],1)
        #sq_0 = comput_squre_1(pi[:,:,0],pi[:,:,1],0)
        #sq_sub = sq_1-sq_0
        y_noise_corrected = road_mutex[uncertain_index]
        #融合纠正后的标签和干净标签
        yc = y[i,:,:,:]
        #y_l = y[i, :, :,: ]
        y1=yc[:,:,1]
        y1[uncertain_index]=y_noise_corrected;
        y0=1-y1
        y[i,:,:,1]=y1;y[i,:,:,0]=y0;
        #将原始标签和纠正标签合并，以供计算损失值
        #yc = np.concatenate((yc,y_l),axis=2)

    return y



if __name__ == '__main__':
    t_label  = 0
    train_data, train_label, test_data, test_label, test_name_list_flat,train_name_list_flat = load_data()
    #train_label = train_label.tolist()
    #train_label=train_label.append(train_label)
    #train_label = np.array(train_label)
    train_label = np.concatenate((train_label,train_label),axis=-1)
    '''
    train_data = train_data[0:76288]
    train_label = train_label[0:76288]
    test_data = test_data[0:10000]
    test_label = test_label[0:10000]
    '''

    #a = np.array(train_data)
    print("len shape:",len(train_data))
    print("len shape:",len(train_label))
    print("len shape:",len(test_data))
    print("len shape:",len(test_label))
    print("lasdasdlasldalsdlasdlaslasdas")
    with tf.device("/gpu:0"):
        if os.path.exists(args.model_path):
            model = NN.FPNNet().build()
            model.load_weights(args.model_path,  by_name=False)

            #model.summary()
            print("checkpoint_loaded success!")
        else:
            model = NN.FPNNet().build()
            #model = t_unet.build(args.sat_size)
            #model.summary()
    multi_gpu_model = multi_gpu_model(model, gpus=[0,1,2,3])
    # multi_gpu_model = model
    #model.summary()
    learning_rate = args.lr
    decay_rate = learning_rate / args.decay_steps
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=True)
    multi_gpu_model.compile(loss=generate_loss,
                  optimizer=sgd
                  )

    for x in model.inputs:
        print("-----------", x.device)
    #labels, names = lpl_prepareData.merge_map_segmentation(test_label, test_name_list_flat)
    callbacks = [
        Metrics(period=args.period, test_data=test_data, test_name_list_flat=test_name_list_flat, model=model,
                labels=test_label)
        # keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, mode='accuracy', save_best_only=False，
        # verbose=1, save_weights_only=False,period=3)
    ]


    # 训练模型
    epoch = 200
    step=1
    #t_label=np.concatenate((train_label,train_label),-1)
    y_correct = train_label
    '''
    print("xxxx:")
    print(train_label.shape)
    print(test_label.shape)
    print(np.array(train_name_list_flat).shape)
    print(np.array(test_name_list_flat).shape)
    print(train_name_list_flat)
    print(test_name_list_flat)
    p_labels,_names = lpl_prepareData.merge_map_segmentation(train_label, train_name_list_flat)
    print("aaaa")
    '''

    for i in range(epoch):
      #if(i==0):
         #t_label = train_label
      print("This is no.%d "%(i))
      #
      #print(train_data[1])
      t_label = y_correct
      multi_gpu_model.fit(x=train_data, y=t_label, callbacks=callbacks, epochs=1,
                        batch_size=args.batchsize, verbose=1, shuffle=True,
                         sample_weight=None)#validation_split=0.00,
      probability_seg = multi_gpu_model.predict(train_data)


      print("probability_seg[0].shape:",probability_seg[0].shape)
      print("probability_seg:",np.array(probability_seg).shape)#5929 128 128 2
      # y_correct = comput_entropy_1(probability_seg,train_label)

      # t_label=y_correct
      '''
      if i >= step:
          #将y_correct与train_name_list_flat对齐
          len_names=len(train_name_list_flat)
          y_shape=y_correct.shape
          y_supplied=np.zeros((len_names-y_shape[0],128,128,4))
          y_correct1=np.concatenate((y_correct,y_supplied),axis=0)
          print("y_correct:",y_correct1.shape)
          print("train_name_list_flat:",len(train_name_list_flat))
          p_labels,_names = lpl_prepareData.merge_map_segmentation(y_correct1, train_name_list_flat)
          path_ = args.image_save +str(i)+'/'
          if not os.path.exists(path_): os.makedirs(path_)
          for j in range (0,len(p_labels)):
              if (_names[j][-4:] != '.tif'):
                  lpl_prepareData.save_image(p_labels[j], path_ + _names[j] + '.tif')
              else:
                  lpl_prepareData.save_image(p_labels[j], path_ + _names[j])
      '''
      #t_label=y*
      #print(test_name_list_flat)

      """
      pre_labels, names = lpl_prepareData.merge_map_segmentation(probability_seg, test_name_list_flat)
      pre_labels = np.array(pre_labels)
      pre_labels.resize(5632,128,128,2)
      t_label = pre_labels
      """

      #print(t_label[1])
      print(" t_label data end!:")
    st = time.time()
    # shuffle=True,
    print('epochs:{}'.format(time.time() - st))
    # 模型评估
    # eval_loss, eval_acc = multi_gpu_model.evaluate(x=test_data, y=test_label, batch_size=args.batchsize)
    time1 = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(time1)
# print('time: {}, Eval loss: {}, Eval Accuracy: {}'.format(time1, eval_loss, eval_acc))
"""def comput_entropy_1(probility,y):
    K = 0.1
    y_update = []
    count = len(probility)
    shape=probility.shape
    pi=np.zeros((shape[1],shape[2],2))

    for i in range(count):
        #分别抽取背景和道路概率
        pi[:,:,0]=probility[i,:, :, 0]
        pi[:, :, 1] = probility[i, :, :, 1]

        #uncertainty map U composed of the uncertainty values of all pixels on an image
        #res0 = tf.Session().run(tf.multiply(pi[:, :, 0], tf.log(pi[:, :, 0])))
        #res1 = tf.Session().run(tf.multiply(pi[:, :, 1], tf.log(pi[:, :, 1])))
        res0 = pi[:, :, 0]*np.log(pi[:, :, 0])
        res1 = pi[:, :, 1]*np.log(pi[:, :, 1])
        u = -(res0+res1)
        #u = tf.Session().run(tf.reduce_sum(res0,res1))
        # 获取噪声标签的索引位置
        #("hello")
        #mean_u = tf.reduce_mean(u,keep_dims=False)
        mean_u = np.mean(u)
        #print(mean_u)
        #print("______________________________________")
        #mean_u = tf.Session().run(tf.reduce_mean(u,keep_dims=False))
        #print(mean_u)
        if(mean_u<K):
            vi = K
        else:
            vi = mean_u
        uncertain_index = np.where(u >= vi)
        #开始纠正标签
        sq_1 = comput_squre_1(pi[:,:,0],pi[:,:,1],1)
        sq_0 = comput_squre_1(pi[:,:,0],pi[:,:,1],0)
        sq_sub = sq_1-sq_0
        y_noise_corrected = np.where(sq_sub[uncertain_index]<0,1,0)
        #融合纠正后的标签和干净标签
        yc = y[i,:,:,:]
        y_l = y[i, :, :,: ]
        y1=yc[:,:,1]
        y1[uncertain_index]=y_noise_corrected;
        y0=1-y1
        yc[:,:,1]=y1;yc[:,:,0]=y0;
        #a = y_l[:,:,0]
        #b = y_l[:,:,1]
        #yc=yc.tolist()
        #yc.append(a)
        #yc.append(b)
        #yc.append(y[i,:,:,0])
        #yc.append(y[i, :, :, 1])

        yc = np.concatenate((yc,y_l),axis=2)

        #yc = np.concatenate((yc,h),axis=2)
        #h = np.resize(1,128,128,4)

        y_update.append(yc)

        #print("aaaa")
        #np.array(y_update)
        #y_update.append(y_l)
        #np.array(y_update)
    return y_update"""