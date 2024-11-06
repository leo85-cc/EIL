# 教育机构：郑州大学
# 学   生：李豪飞
# 开发时间： 15:47
import numpy as np
import torch
from einops import rearrange, repeat
import random
import torch.nn as nn
from model import unet_model
'''load other file'''
import sys
sys.path.append("/public/home/zzuegr01/lhf/MUL/CBRNet-main_ex")
from standard import lpl_logger
import lpl_prepareData as preparData
# from standard import lpl_accuracy as acc
from torchsummary import summary
import lpl_accuracy1 as lpl_accuracy
import os
import time
import argparse
from mpi4py import MPI

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
str_rank = str(rank)


def Create_args():
    parser = argparse.ArgumentParser()
    # Training settings
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--sat_size', type=int, default=128)
    parser.add_argument('--sat_channel', type=int, default=3)
    parser.add_argument('--map_size', type=int, default=128)
    parser.add_argument('--map_channel', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    # 学习参数
    parser.add_argument('--lr', type=float, default=1e-4)  # 设置学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)  # 设置衰减率
    parser.add_argument('--decay_steps', type=float, default=20000)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # 通道尺寸
    parser.add_argument('--batchsize_pro', type=int, default=128)
    parser.add_argument('--cut_size', type=int, default=16)
    parser.add_argument('--in_channels', type=int, default=1024)
    parser.add_argument('--out_channels', type=int, default=2)
    # 获取数据
    parser.add_argument('--image_db', type=str, default=
    '/public/home/zzuegr01/zt/core_dataset/massa_building/train/sat_s')
    # '/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/train/sat_s')
    parser.add_argument('--label_db', type=str, default=
    '/public/home/zzuegr01/zt/core_dataset/massa_building/train/map_s')
    # '/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/train/sat_s')
    parser.add_argument('--test_image_db', type=str, default=
    '/public/home/zzuegr01/zt/core_dataset/massa_building/test/sat_s')
    # '/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/test/sat_s')
    parser.add_argument('--test_label_db', type=str, default=
    '/public/home/zzuegr01/zt/core_dataset/massa_building/test/map_s')
    # '/public/home/zzuegr01/zt/core_dataset/phoenix_dataset/test/map_s')
    #保存路径
    # 保存路径
    parser.add_argument('--model_save_path', type=str, default='./log_/save_model/process_' + str_rank + '/')
    parser.add_argument('--log_path', type=str, default='./log_/log/process_' + str_rank + '/')
    parser.add_argument('--resule_path', type=str, default='./log_/image/process_' + str_rank + '/')
    parser.add_argument('--load_model', type=str,
                        default='/public/home/zzuegr01/lhf/Experimeat_02/CBRNet/model/model_1.pth')
    # _1e-09_1.0_6.5_0.8_4_202109101846_model_at_epoch_9.h5')
    # ~/ljm/lpl_best_model/1e-09_1.0_5.0_202104060302_model_at_epoch_18.h5
    parser.add_argument('--ratio_ro', type=float, default='1.5')
    parser.add_argument('--ratio_bu', type=float, default='1')
    parser.add_argument('--ratio_bg', type=float, default='1')
    parser.add_argument('--ratio_no', type=float, default='1')
    parser.add_argument('--lam', type=float, default='1')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--optimizer_flag', type=int, default=1)
    # 实例化
    args = parser.parse_args()
    return args


# 参数实例化
args = Create_args()
print("--lr-- ", args.lr)
# 若文件夹不存在则创建，在多进程中不使用try容易报错
def make_file(file):
    if not os.path.exists(file):
        try:
            os.makedirs(file)
        except OSError:
            pass
make_file(args.resule_path)
make_file(args.log_path)
make_file(args.model_save_path)

# 设置保存路径和文件名
common_name = os.environ['SLURM_JOBID'] + '_' + str(args.lr) + '_' + str(args.batchsize_pro) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro)+ '_unet'
acc_logger = lpl_logger.logger(args.log_path,
                               common_name)

if args.resule_path is not None:
    resule_path = args.resule_path + common_name + '_' + str(args.lr) + '_' + str(args.ratio_ro) + '_' + str(args.ratio_bg) + '/'
    make_file(resule_path)


# 函数设置
def load_data():
    batchs_data, batchs_label, key = preparData.GetData(args, args.image_db, args.label_db)
    test_data, test_label, test_name_list = preparData.GetData(args, args.test_image_db, args.test_label_db)
    test_name_list_flat = []
    # 对影像快名字进行flat,二维变一维
    for i in range(len(test_name_list)):
        for j in range(len(test_name_list[i])):
            test_name_list_flat.append(test_name_list[i][j])
    return batchs_data, batchs_label, test_data, test_label, test_name_list_flat


'''合成图片+精度预测'''


def printf_acc(iter_times, probability_seg, test_name_list, test_label, num, path=args.resule_path):
    print('printf acc...')
    _test_label = []
    pre_ro_sum = 0
    rec_ro_sum = 0
    f1_ro_sum = 0
    pre_labels, names = preparData.merge_map_segmentation(probability_seg, test_name_list)
    for i in range(0, len(test_label)):
        test_data_sub = test_label[i]
        for test_index in range(0, len(test_data_sub)):
            test_data_sub_item = test_data_sub[test_index, :, :, :]
            new_shape = (1,) + test_data_sub_item.shape
            test_data_sub_item = np.reshape(test_data_sub_item, new_shape)
            _test_label.append(test_data_sub_item)
    labels, names = preparData.merge_map_segmentation(_test_label, test_name_list, 3)
    avg_TP = 0;avg_FP = 0;avg_FN = 0;
    for i in range(0, len(pre_labels)):
        pre_label = pre_labels[i]
        label = labels[i]
        TP, FP, FN = lpl_accuracy.acc_2D(pre_label, label)
        avg_TP = avg_TP + TP
        avg_FP = avg_FP + FP
        avg_FN = avg_FN + FN
        
    precision = float(avg_TP) / float(avg_TP + avg_FP + 0.0001)
    recall = float(avg_TP) / float(avg_TP + avg_FN + 0.0001)
    ave_f1 = lpl_accuracy.acc_f1(precision, recall)
    iou = float(avg_TP) / float(avg_TP + avg_FP + avg_FN + 0.0001)
    acc_logger.output([iter_times, precision, recall, ave_f1, iou])
    print(iter_times, precision, recall, ave_f1, iou)
    return precision, recall, ave_f1


'''操作合成图片'''

def pro_seg(batchs_data):
    list1 = []
    index = np.arange(len(batchs_data))
    count_in = 0
    list_data = []
    for i in index:
        list_data.append(batchs_data[i])
        count_in += 1
        if count_in % args.batchsize_pro == 0 or count_in == len(batchs_data):
            train_data = np.array(list_data)
            train_data = torch.from_numpy(train_data).float().cuda()
            input_data = rearrange(train_data, 'l b h w c -> (l b) c h w')  # [56074, 3, 64, 64]
            '''输入模型'''
            pre_out = model(input_data)
            L = rearrange(pre_out, 'k c h w -> k h w c')
            L = L.cpu()  # 转换到cpu
            # pre_out1 = pre_out.detach().numpy()
            # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
            # precision[i] = predict_num
            L = L.detach().numpy()
            for j in range(L.shape[0]):
                list1.append(L[j, :, :, :])
            list_data = []
    return list1


def acc1(test_data, test_name_list_flat, test_label,num):
    list1 = []
    index = np.arange(len(test_data))
    count_in = 0
    list_data = []
    for i in index:
        list_data.append(test_data[i])
        count_in += 1
        if count_in % args.batchsize_pro == 0 or count_in == len(test_data):
            train_data = np.array(list_data)
            train_data = torch.from_numpy(train_data).float().cuda()
            input_data = rearrange(train_data, 'l b h w c -> (l b) c h w')  # [56074, 3, 64, 64]
            '''输入模型'''
            pre_out = model(input_data)
            L = rearrange(pre_out, 'k c h w -> k h w c')
            L = L.cpu()  # 转换到cpu
            # pre_out1 = pre_out.detach().numpy()
            # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
            # precision[i] = predict_num
            L = L.detach().numpy()
            L = L[:, np.newaxis, :, :, :]
            for j in range(L.shape[0]):
                list1.append(L[j, :, :, :, :])
            list_data = []
    pre, rec, ave = printf_acc(1, list1, test_name_list_flat, test_label, num)
    return pre, rec, ave


'''取最大'''
def max_acc(i):
    if args.model_save_path is not None:
        make_file(args.model_save_path + common_name + '_' + str(args.lr) + '_' + str(args.ratio_ro) + '_' + str(args.ratio_bg) + '/')
        path = args.model_save_path + common_name + '_' + str(args.lr) + '_' + str(args.ratio_ro) + '_' + str(args.ratio_bg) + '/' + 'model_' + str(i) + '.pth'
        # torch.save(model.state_dict(), path)
        torch.save(model.module.state_dict(), path)


def regu_tensor2(p1, p2):
    # 提取目标（道路）的tensor
    extend_p1 = torch.unsqueeze(p1, 3)
    extend_p2 = torch.unsqueeze(p2, 3)
    P = torch.cat([extend_p1, extend_p2], 3)
    # 计算方差
    # meanP = torch.mean(P, 3)
    varP = torch.std(P, 3)
    max_varP = torch.max(varP)
    # 获取噪声索引
    one = torch.ones_like(p1)
    noisy1 = torch.where(varP > args.ratio_no * max_varP, p1, one)
    noisy2 = torch.where(varP > args.ratio_no * max_varP, p2, one)
    regu = -torch.sum(noisy1 * torch.log(noisy1) + noisy2 * torch.log(noisy2))
    return regu
# 计算损失值
def generate_loss(y_true, y_pred):
    # ce
    res0 = -torch.sum(torch.mul(y_true[:, :, :, args.T], torch.log(torch.add(1e-7, y_pred[:, :, :, 0]))))
    res1 = -torch.sum(torch.mul(y_true[:, :, :, args.T + 1], torch.log(torch.add(1e-7, y_pred[:, :, :, 1]))))
    regu = regu_tensor2(y_true[:, :, :, 0], y_true[:, :, :, 1])
    res = (1 - args.lam) * torch.sum(args.ratio_bg * res0 + args.ratio_ro * res1) + args.lam * regu
    # lpl_tensorboard.variable_summaries(prob[:, :, :, 1])
    return res

def comput_squre_2(a, b, n):
    sum_0 = (n - a) ** 2 + (n - b) ** 2
    return sum_0

def comput_entropy_2(p1, p2, y):
    y_update = []
    count = len(p1)
    shape = p1.shape
    pi = np.zeros((shape[1], shape[2], 2))
    for i in range(count):
        # 抽取道路通道
        pi[:, :, 0] = p1[i, :, :, 0]
        pi[:, :, 1] = p2[i, :, :, 0]
        # 获取道路概率的方差
        e = np.var(pi, 2)
        max_v = np.max(e)
        # 获取噪声标签的索引位置
        noise_index = np.where(e >= args.ratio_no * max_v)
        # 开始纠正标签
        sq_1 = comput_squre_2(pi[:, :, 0], pi[:, :, 1], 1)
        sq_0 = comput_squre_2(pi[:, :, 0], pi[:, :, 1], 0)
        sq_sub = sq_1 - sq_0
        y_noise_corrected = np.where(sq_sub[noise_index] < 0, 1, 0)
        # 融合纠正后的标签和干净标签
        yc = y[i, :, :, :]
        y1 = yc[:, :, 1]
        y1[noise_index] = y_noise_corrected
        y0 = 1 - y1
        yc[:, :, 1] = y1
        yc[:, :, 0] = y0
        y_update.append(yc)
    return y_update
# 主函数设置

# 获取数据
batchs_data, batchs_label, test_data, test_label, test_name_list_flat = load_data()
batchs_label1 = np.array(batchs_label)
batchs_label1 = np.squeeze(batchs_label1,1)
# 生成日志
# acc_logger = lpl_logger.logger('./two_amap_gps__logs/', 'zz_amap_gps__s_' + str(0.2) + '_' + str(1) + '_' + str(1))
# 将模型放入DCU
# demo4是模型的重点部分
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = unet_model.CBRNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
'''加载模型参数'''
model.module.load_state_dict(torch.load(args.load_model))
'''精确度计算'''
jishu = 0
start = time.time()
pre = 0
rec = 0
f1 = 0
# max = max_acc(0)
pre,rec,f1 = acc1(test_data,test_name_list_flat,test_label,jishu)
end = time.time()
print("合成图片时间为：", end - start)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train_loss = []
test_loss = []

current_labels = batchs_label
sdl = []  # 标签概率队列

count = 0
acc = []
max = 0

probability_seg = pro_seg(batchs_data)
probability_seg = np.array(probability_seg)
print('完成')
mul_labels = probability_seg[:, :, :, 1, None]
for i in range(args.T - 1):
    mul_labels = np.concatenate((mul_labels, probability_seg[:, :, :, 1, None]), -1)
mul_labels = np.concatenate((mul_labels, batchs_label1[:, :, :, 0, None]), -1)
mul_labels = np.concatenate((mul_labels, batchs_label1[:, :, :, 1, None]), -1)

for item in range(args.decay_steps):

    if args.T == 2:
        if len(sdl) <= args.T - 1:
            sdl.append(probability_seg[:, :, :, 1, None])
        else:
            # 删除最早的预测标签概率
            sdl.pop(0)
            # 添加当前预测标签概率
            sdl.append(probability_seg[:, :, :, 1, None])
            # 纠正当前标签
            corrrect_label = comput_entropy_2(sdl[0], sdl[1], batchs_label1)
            current_labels = np.array(corrrect_label)
            sdl.append(current_labels[:, :, :, 0, None])
            sdl.append(current_labels[:, :, :, 1, None])
            # 更新多重训练标签
            mul_labels = np.array(sdl[0])
            for j in range(1, len(sdl)):
                item = np.array(sdl[j])
                mul_labels = np.concatenate((mul_labels, item), -1)
            sdl.pop()
            sdl.pop()
            print("计算完成")

    start = time.time()
    test_loss_sum = 0
    index = np.arange(len(batchs_data))
    np.random.shuffle(index)
    count_in = 0
    iter_times = 0
    list_data = []
    list_label = []
    for i in index:
        train_loss_sum = 0
        list_data.append(batchs_data[i])
        list_label.append(mul_labels[i,:,:,:])
        count_in += 1
        if count_in % args.batchsize_pro == 0 or count_in == len(batchs_data):
            train_data = np.array(list_data)
            train_label = np.array(list_label)
            train_data = torch.from_numpy(train_data).float().cuda()
            train_label = torch.from_numpy(train_label).float().cuda()
            input_data1 = rearrange(train_data, 'l b h w c -> (l b) c h w')  # [56074, 3, 64, 64]
            # label_data1 = rearrange(train_label, 'b h w c -> b h w c')  # [56074, 2, 64, 64]
            '''输入模型'''
            k1 = model(input_data1)  # [529, 2, 64, 64])
            k1 = rearrange(k1, 'k c h w -> k h w c')  # [529, 64, 64, 2]
            loss = generate_loss(train_label,k1)
            optimizer.zero_grad()  # 对梯度的值要进行清零
            loss.backward()
            # 完成梯度下降的操作w = w - dw
            optimizer.step()
            iter_times = iter_times + 1
            list_data = []
            list_label = []
            print("{0}---{1}---".format(item, iter_times), loss, pre, rec, f1)
    print('计算开始')
    probability_seg = pro_seg(batchs_data)
    probability_seg = np.array(probability_seg)
    print('计算结束')
    jishu += 1
    end = time.time()
    print("第{0}次时间为：".format(item), end - start)
    pre, rec, f1 = acc1(test_data, test_name_list_flat, test_label,jishu)
    acc.append(f1)
    max = max_acc(jishu)