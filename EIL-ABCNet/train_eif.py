import numpy as np
import torch
from einops import rearrange, repeat
import random
from model import ABCNet
import torch.nn as nn
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
    parser.add_argument('--ratio_bg', type=float, default=1)
    parser.add_argument('--ratio_ro', type=float, default=2)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--a0', type=float, default=0.1)
    parser.add_argument('--a1', type=float, default=2)

    parser.add_argument('--lr', type=float, default=1e-4)  # 设置学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)  # 设置衰减率
    parser.add_argument('--decay_steps', type=float, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

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

    parser.add_argument('--model_save_path', type=str, default='./mul/model_param/process_' + str_rank + '/')
    parser.add_argument('--log_path', type=str, default='./mul/two_amap_gps__logs/process_' + str_rank + '/')
    parser.add_argument('--result_path', type=str, default='./mul/Result/process_' + str_rank + '/')
    parser.add_argument('--load_model', type=str, default='./model_list0/model_' + str_rank + '/')
    # '/public/home/zhaoqb/lhf/CBRNet-main/model_param/checkpoint_net-0.0001-221.pth')
    args = parser.parse_args()
    return args


# 参数实例化
args = Create_args()


def make_file(file):
    if not os.path.exists(file):
        try:
            os.makedirs(file)
        except OSError:
            pass


make_file(args.model_save_path)
make_file(args.log_path)
make_file(args.result_path)

common_name = os.environ['SLURM_JOBID'] + '_' + str(args.lr) + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro)


# 函数设置
def load_data():
    batchs_data, batchs_label, key = preparData.GetData(args, args.image_db, args.label_db)
    test_data, test_label, test_name_list = preparData.GetData(args, args.test_image_db, args.test_label_db)
    test_name_list_flat = []
    # 对影像快名字进行flat,二维变一维
    for i in range(len(test_name_list)):
        for j in range(len(test_name_list[i])):
            test_name_list_flat.append(test_name_list[i][j])
    batchs_data = np.asarray(batchs_data).squeeze()
    batchs_label = np.asarray(batchs_label).squeeze()
    test_data = np.asarray(test_data).squeeze()
    test_label = np.asarray(test_label).squeeze()
    return batchs_data, batchs_label, test_data, test_label, test_name_list_flat


'''合成图片+精度预测'''


def printf_acc(iter_times, test_name_list, test_label, path=args.result_path):
    test_seg = []
    index = np.arange(len(test_data))
    count_in = 0
    list_data = []
    list_label = []
    for i in index:
        train_loss_sum = 0
        list_data.append(test_data[i])
        count_in += 1
        if count_in % args.batchsize_pro == 0 or count_in == len(test_data):
            train_data = np.array(list_data)
            train_data = torch.from_numpy(train_data).float().cuda()
            input_data = rearrange(train_data, 'l h w c -> l c h w')  # [56074, 3, 64, 64]
            '''输入模型'''
            seg = model(input_data)
            L = rearrange(seg, 'k c h w -> k h w c')
            L = L.cpu()  # 转换到cpu
            # pre_out1 = pre_out.detach().numpy()
            # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
            # precision[i] = predict_num
            L = L.detach().numpy()
            L = L[:, np.newaxis, :, :, :]
            for j in range(L.shape[0]):
                test_seg.append(L[j, :, :, :, :])
            list_data = []
            list_label = []
    train_seg = []
    count_in = 0
    list_data = []
    list_label = []
    for i in index:
        train_loss_sum = 0
        list_data.append(batchs_data[i])
        count_in += 1
        if count_in % args.batchsize_pro == 0 or count_in == len(batchs_data):
            train_data = np.array(list_data)
            train_data = torch.from_numpy(train_data).float().cuda()
            input_data = rearrange(train_data, 'l h w c -> l c h w')  # [56074, 3, 64, 64]
            '''输入模型'''
            seg = model(input_data)
            L = rearrange(seg, 'k c h w -> k h w c')
            L = L.cpu()  # 转换到cpu
            # pre_out1 = pre_out.detach().numpy()
            # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
            # precision[i] = predict_num
            L = L.detach().numpy()
            L = L[:, np.newaxis, :, :, :]
            for j in range(L.shape[0]):
                train_seg.append(L[j, :, :, :, :])
            list_data = []
            list_label = []
    probability_test_seg = np.zeros([len(test_data), 128, 128, 2], dtype=np.float16)

    t0 = time.time()

    probability_seg = np.asarray(train_seg, np.float16)

    t1 = time.time()
    print('成功获取预测结果!', t1 - t0)

    dim0 = probability_seg.shape[0]
    # 存放所有数据
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

    print(all_seg.shape)

    t2 = time.time()
    print('train_data传递成功！', t2 - t1)
    step = 1024
    batchs = len(probability_seg) // step
    # print(batchs)

    for index in range(batchs // 3):
        # print(index)
        i = random.randint(0, batchs - 1)
        # print(i)
        a0 = args.a0
        a1 = args.a1
        # pro = p0 = p1 = 0

        var = np.zeros([size + 1, step, 128, 128], dtype=np.float16)
        # vartest = np.zeros([4, 128, 128])
        for j in range(size):
            var[j] = all_seg[j, i * step:i * step + step, :, :, 1]
        var[size] = batchs_label[i * step:i * step + step, :, :, 1]
        for j in range(size + 1):
            pro = var[j, :, :, :]
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
        for j in range(size + 1):
            pro = var[j, :, :, :]
            p0 = 1 - pro
            p1 = pro
            r0 = 1 / (1 + np.exp(-a0 * p0))
            r1 = 1 / (1 + np.exp(-a1 * p1))
            rc0 = rc0 + r0 * p0
            rc1 = rc1 + r1 * p1

        index_bulding = np.where(rc1 >= rc0)
        y = np.zeros([step, 128, 128])
        y[index_bulding] = 1
        x = 1 - y
        # 构建新标签集
        newlabels[i * step:i * step + step, :, :, 0] = x
        newlabels[i * step:i * step + step, :, :, 1] = y

    t3 = time.time()
    print('train_data计算成功!', t3 - t2)

    probability_seg = np.asarray(test_seg, np.float16)

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

    for i in range(len(probability_seg)):
        a0 = 0.1
        a1 = 2
        # pro = p0 = p1 = 0

        var = np.zeros([size, 128, 128], dtype=np.float16)
        # vartest = np.zeros([4, 128, 128])
        for j in range(size):
            var[j] = all_seg[j, i, :, :, 1]
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
        probability_test_seg[i, :, :, 0] = x
        probability_test_seg[i, :, :, 1] = y
    # print(self.probability_test_seg)
    t6 = time.time()
    print('test_data计算完成!', t6 - t5)

    pre_labels, names = preparData.merge_map_segmentation(probability_test_seg, test_name_list_flat)

    pre_ro_sum = 0
    rec_ro_sum = 0
    f1_ro_sum = 0
    labels, names = preparData.merge_map_segmentation(test_label, test_name_list, 3)

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


def acc1(iter_times, test_name_list_flat, test_label):
    # test1 = []
    # index = np.arange(len(test_data))
    # count_in = 0
    # list_data = []
    # list_label = []
    # for i in index:
    #     train_loss_sum = 0
    #     list_data.append(test_data[i])
    #     count_in += 1
    #     if count_in % args.batchsize_pro == 0 or count_in == len(test_data):
    #         train_data = np.array(list_data)
    #         train_data = torch.from_numpy(train_data).float().cuda()
    #         input_data = rearrange(train_data, 'l h w c -> l c h w')  # [56074, 3, 64, 64]
    #         '''输入模型'''
    #         seg = model(input_data)
    #         L = rearrange(seg, 'k c h w -> k h w c')
    #         L = L.cpu()  # 转换到cpu
    #         # pre_out1 = pre_out.detach().numpy()
    #         # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
    #         # precision[i] = predict_num
    #         L = L.detach().numpy()
    #         L = L[:, np.newaxis, :, :, :]
    #         for j in range(L.shape[0]):
    #             test1.append(L[j, :, :, :, :])
    #         list_data = []
    #         list_label = []
    # train1 = []
    # index = np.arange(len(batchs_data))
    # count_in = 0
    # list_data = []
    # list_label = []
    # for i in index:
    #     train_loss_sum = 0
    #     list_data.append(batchs_data[i])
    #     count_in += 1
    #     if count_in % args.batchsize_pro == 0 or count_in == len(batchs_data):
    #         train_data = np.array(list_data)
    #         train_data = torch.from_numpy(train_data).float().cuda()
    #         input_data = rearrange(train_data, 'l h w c -> l c h w')  # [56074, 3, 64, 64]
    #         '''输入模型'''
    #         seg = model(input_data)
    #         L = rearrange(seg, 'k c h w -> k h w c')
    #         L = L.cpu()  # 转换到cpu
    #         # pre_out1 = pre_out.detach().numpy()
    #         # predict_num = demo4.predict(pre_out1,label_data[n,:,:,:,:])
    #         # precision[i] = predict_num
    #         L = L.detach().numpy()
    #         L = L[:, np.newaxis, :, :, :]
    #         for j in range(L.shape[0]):
    #             train1.append(L[j, :, :, :, :])
    #         list_data = []
    #         list_label = []
    pre, rec, ave = printf_acc(iter_times, test_name_list_flat, test_label)
    return pre, rec, ave


'''取最大'''


def max_acc(item, acc, max):
    for i in range(len(acc)):
        if acc[i] > max:
            make_file(args.model_save_path + common_name + '/')
            path = args.model_save_path + common_name + '/' + 'checkpoint_net' + '_' + str(args.lr) + '_' + str(
                args.ratio_bg) \
                   + '_' + str(args.ratio_ro) + '_epoch_' + str(item) + '.pth'
            # torch.save(model.state_dict(), path)
            torch.save(model.module.state_dict(), path)
            max = acc[i]
    return max


# 计算损失值
def generate_loss(prob, data_o, data_n):
    # ce
    res1 = -torch.sum(torch.mul(data_o[:, :, :, 0], torch.log(torch.add(1e-7, prob[:, :, :, 0]))))
    res2 = -torch.sum(torch.mul(data_o[:, :, :, 1], torch.log(torch.add(1e-7, prob[:, :, :, 1]))))
    res3 = -torch.sum(torch.mul(data_n[:, :, :, 0], torch.log(torch.add(1e-7, prob[:, :, :, 0]))))
    res4 = -torch.sum(torch.mul(data_n[:, :, :, 1], torch.log(torch.add(1e-7, prob[:, :, :, 1]))))
    loss_o = torch.sum(args.ratio_bg * res1 + args.ratio_ro * res2)
    loss_n = torch.sum(args.ratio_bg * res3 + args.ratio_ro * res4)
    # lpl_tensorboard.variable_summaries(prob[:, :, :, 1])
    return loss_o + args.lam * loss_n
# 主函数设置


# 获取数据
batchs_data, batchs_label, test_data, test_label, test_name_list_flat = load_data()
newlabels = np.array(batchs_label, dtype=batchs_label.dtype)
# 生成日志
acc_logger = lpl_logger.logger(args.log_path, os.environ['SLURM_JOBID'] + '_' + str(args.lr)
                               + '_' + str(args.ratio_bg) + '_' + str(args.ratio_ro))
# 将模型放入DCU
# demo4是模型的重点部分
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3.4'
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = ABCNet.ABCNet(3, 2)
model = torch.nn.DataParallel(model)
model = model.cuda()
'''加载模型参数'''

model_list = [os.path.join(args.load_model, name) for name in os.listdir(args.load_model)]
if model_list:
    latest_model = max(model_list, key=os.path.getctime)
    model.module.load_state_dict(torch.load(latest_model))
    print('进程号:' + str_rank, '---------------------成功加载模型参数:', latest_model, '---------------------------')
else:
    print('进程号:' + str_rank, '---------------------未加载模型!------------------------')

'''精确度计算'''
iter_times = 0
start = time.time()
pre, rec, f1 = acc1(0, test_name_list_flat, test_label)
end = time.time()
print("rank:", rank, "合成图片时间为：", end - start)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train_loss = []
test_loss = []

count = 0
acc = []
max = 0
# 每epoch
for item in range(args.decay_steps):
    start = time.time()
    test_loss_sum = 0
    index = np.arange(len(batchs_data))
    np.random.shuffle(index)
    count_in = 0
    iter_times = 0
    list_data = []
    list_label = []
    list_new = []
    # 凑够批次大小128
    for i in index:
        train_loss_sum = 0
        list_data.append(batchs_data[i])
        list_label.append(batchs_label[i])
        list_new.append(newlabels[i])
        count_in += 1
        # 没批次
        if count_in % args.batchsize_pro == 0 or count_in == len(batchs_data):
            train_data = np.array(list_data)
            train_label = np.array(list_label)
            train_new = np.array(list_new)
            train_data = torch.from_numpy(train_data).float().cuda()
            train_label = torch.from_numpy(train_label).float().cuda()
            train_new = torch.from_numpy(train_new).float().cuda()
            input_data1 = rearrange(train_data, 'l h w c -> l c h w')  # [56074, 3, 64, 64]
            label_data1 = rearrange(train_label, 'l h w c -> l h w c')  # [56074, 2, 64, 64]
            new_data1 = rearrange(train_new, 'l h w c -> l h w c')
            '''输入模型'''
            k1 = model(input_data1)  # [529, 2, 64, 64])
            k1 = rearrange(k1, 'k c h w -> k h w c')  # [529, 64, 64, 2]
            loss = generate_loss(k1, label_data1, new_data1)
            optimizer.zero_grad()  # 对梯度的值要进行清零
            loss.backward()
            # 完成梯度下降的操作w = w - dw
            optimizer.step()
            iter_times = iter_times + 1
            list_data = []
            list_label = []
            list_new = []
            if rank == 0:
                print("{0}---{1}---".format(item, iter_times), loss, pre, rec, f1)
    end = time.time()
    if rank == 0:
        print("第{0}次时间为：".format(item), end - start)
    pre, rec, f1 = acc1(item, test_name_list_flat, test_label)
    acc.append(f1)
    max = max_acc(item, acc, max)