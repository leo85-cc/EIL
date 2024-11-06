'''lpl 19/07/02
精确度评价指标'''
'''拼接后的影像精确度评价指标'''
import numpy as np
import auxiliary
#批量计算模型的精确度，输入为模型的输出和标签
def acc_2D_batch(pre_pro,labels):
    sum_TP=0;sum_FP=0;sum_FN=0
    for i in range(0, len(pre_pro)):
        #将预测结果还原为标签
        pre_label = auxiliary.convert_labels2C_inverse(pre_pro[i])
        label = labels[i,:,:,1]*255
        # 使用矩阵运算对精确度计算进行加速

        prediction = np.array(pre_label, dtype=np.float)
        label = np.array(label, dtype=np.float)
        #iou=acc_iou(prediction,label)
        # 计算TP
        multiple = prediction * label
        TP = np.sum(np.sum(multiple / (255 * 255)))
        sub = prediction - label
        FP = sum(sum(np.where(sub > 0, sub, 0))) / 255
        FN = -sum(sum(np.where(sub < 0, sub, 0))) / 255
        #累加 TP,FP,FN
        sum_TP = sum_TP + TP
        sum_FP = sum_FP + FP
        sum_FN = sum_FN + FN
    if sum_TP + sum_FP!=0:
        precision = float(sum_TP) / float(sum_TP + sum_FP)
    else:
        precision=0
    if sum_TP + sum_FN!=0:
        recall = float(sum_TP) / float(sum_TP + sum_FN)
    else:
        recall=0
    if sum_TP+sum_FP+sum_FN!=0:
        iou=float(sum_TP) / float(sum_TP+sum_FP+sum_FN)
    else:
        iou=0
    return precision, recall,iou
def acc_2D_batch_(pre_pro,labels):
    sum_TP=0;sum_FP=0;sum_FN=0
    for i in range(0, len(pre_pro)):
        #将预测结果还原为标签
        pre_label = auxiliary.convert_labels2C_inverse(pre_pro[i])
        label = labels[i,:,:,1]*255
        # 使用矩阵运算对精确度计算进行加速
        prediction = np.array(pre_label, dtype=np.float)
        label = np.array(label, dtype=np.float)
        # 计算TP
        multiple = prediction * label
        TP = np.sum(np.sum(multiple / (255 * 255)))
        sub = prediction - label
        FP = sum(sum(np.where(sub > 0, sub, 0))) / 255
        FN = -sum(sum(np.where(sub < 0, sub, 0))) / 255
        #累加 TP,FP,FN
        sum_TP = sum_TP + TP
        sum_FP = sum_FP + FP
        sum_FN = sum_FN + FN
    if sum_TP + sum_FP!=0:
        precision = float(sum_TP) / float(sum_TP + sum_FP)
    else:
        precision=0
    if sum_TP + sum_FN!=0:
        recall = float(sum_TP) / float(sum_TP + sum_FN)
    else:
        recall=0
    return precision, recall
def acc_2D(prediction,label):
    #使用矩阵运算对精确度计算进行加速
    prediction=np.array(prediction,dtype=np.float)
    label = np.array(label, dtype=np.float)
    precision=0; recall=0
    #计算TP
    multiple=prediction*label
    TP=np.sum(np.sum(multiple/(255*255)))
    sub=prediction-label
    FP = sum(sum(np.where(sub > 0, sub, 0))) / 255
    FN = -sum(sum(np.where(sub < 0, sub, 0))) / 255
    #print('Optimal ',' TP: ', TP, ' FP: ', FP, ' FN: ', FN)
    if TP + FP!=0:
        precision = float(TP) / float(TP + FP)
    else:
        precision=0
    if TP + FN!=0:
        recall = float(TP) / float(TP + FN)
    else:
        recall=0
    return precision, recall
def acc_2D_old(prediction,label):
    shape = label.shape
    TP = 0;FP = 0
    FN = 0;TN = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            if label[i][j]!=0 and prediction[i][j]!=0:
                TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
    if TP + FP!=0:
        precision = float(TP) / float(TP + FP)
    else:
        precision=0
    if TP + FN!=0:
        recall = float(TP) / float(TP + FN)
    else:
        recall=0
    return precision, recall
def acc_2D_2_class(prediction,label):
    shape = label.shape
    TP = 0;FP = 0
    FN = 0;TN = 0
    TP1 = 0;FP1 = 0
    FN1 = 0;TN1 = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            if label[i][j]!=0 and prediction[i][j]!=0:
                TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
            if label[i][j]!=255 and prediction[i][j]!=255:
                TP1+=1
            elif label[i][j]!=255 and prediction[i][j]==255:
                FN1+=1
            elif label[i][j]==255 and prediction[i][j]!=255:
                FP1+=1
            elif label[i][j]==255 and prediction[i][j]==255:
                TN1+=1
    if TP + FP!=0:
        precision = float(TP) / float(TP + FP)
    else:
        precision=0
    if TP + FN!=0:
        recall = float(TP) / float(TP + FN)
    else:
        recall=0
    if TP1 + FP1 != 0:
        precision1 = float(TP1) / float(TP1 + FP1)
    else:
        precision = 0
    if TP1 + FN1 != 0:
        recall1 = float(TP1) / float(TP1 + FN1)
    else:
        recall = 0
    return precision, recall,precision1, recall1
'''拼接后的影像精确度_relx评价指标'''
def acc_2D_relx(prediction,label,N=3):
    shape = label.shape
    TP = 0;FP = 0;TP_relax=0
    FN = 0;TN = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            #if IsRight_N(label, i, j, 255, N) and prediction[i][j] != 0:
               # TP_relax = TP_relax + 1
            if label[i][j]!=0 and prediction[i][j]!=0:
                TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
    if TP_relax + FP!=0:
        pre = float(TP_relax) / float(TP_relax + FP)
    else:
        pre = 0
    if TP_relax + FN != 0:
        rec = float(TP_relax) / float(TP_relax + FN)
    else:
        rec = 0
    if pre + rec != 0:
        f1 = 2 * pre * rec / (pre + rec)
    else:
        f1 = 0
    #print(TP_relax, FP, FN)

    return pre, rec
def acc_2D_relx_iou(prediction,label,N=3):
    shape = label.shape
    TP = 0;FP = 0;TP_relax=0
    FN = 0;TN = 0
    for i in  range (0,shape[0]):
        for j in range(0,shape[1]):
            if IsRight_N(label, i, j, 255, N) and prediction[i][j] != 0:
                TP_relax = TP_relax + 1
            if label[i][j]!=0 and prediction[i][j]!=0:
                TP+=1
            elif label[i][j]!=0 and prediction[i][j]==0:
                FN+=1
            elif label[i][j]==0 and prediction[i][j]!=0:
                FP+=1
            elif label[i][j]==0 and prediction[i][j]==0:
                TN+=1
    if TP_relax + FP!=0:
        pre = float(TP_relax) / float(TP_relax + FP)
    else:
        pre = 0
    if TP_relax + FN != 0:
        rec = float(TP_relax) / float(TP_relax + FN)
    else:
        rec = 0
    if pre + rec != 0:
        f1 = 2 * pre * rec / (pre + rec)
    else:
        f1 = 0
    #print(TP_relax, FP, FN)
    iou=TP_relax/(TP_relax+FP+FN)
    return pre, rec,iou
'''判断预测像素的临近范围内是否有相应对的标签。邻域大小为该像素的上、小、左、右个N像素'''
def IsRight_N(label,pos_row,pos_column,key_pixel,N):
    label_shape=label.shape
    start_row=0 if pos_row-N<0 else pos_row -N
    end_row=label_shape[0]-1 if pos_row+N>=label_shape[0] else pos_row+N
    start_column= 0 if pos_column-N<0 else pos_column-N
    end_column=label_shape[1]-1 if pos_column+N>=label_shape[1] else pos_column+N
    # '''corss shape'''
    # for i in range(start_row,end_row+1):
    #     if label[i][pos_column]==key_pixel:
    #         return True
    # for j in range(start_column,end_column+1):
    #     if label[pos_row][j]==key_pixel:
    #         return True
    #田字形缓冲区
    for i in range(start_row,end_row+1):
        for j in range(start_column,end_column+1):
            if label[i][j]==key_pixel:
                return True
    return False
'''计算单类别预测目标的精确度，比如，道路和建筑物，可以同时计算两个目标的精确度。标签形式为3D-矩阵，
矩阵中的每个元素是一个三维的向量(v1,v2，v3)。v1代表该像素属于背景的的概率，v2代表属于预测目标1的概率，
v3代表属于预测目标2的概率。
已改动'''
# def acc_3D(pre,label,key_pixel1=255,key_pixel2=76,key_pixel3=0):
#     shape=label.shape
#     TP_1=0;FP_1=0;FN_1=0;TN_1=0
#     TP_2 = 0; FP_2 = 0; FN_2 = 0; TN_2 = 0
#     TP_3 = 0; FP_3 = 0; FN_3 = 0; TN_3 = 0
#     for i in range(0, shape[0]):
#         for j in range(0, shape[1]):
#             # 目标1的精度
#             if label[i][j] == key_pixel1 and pre[i][j] == key_pixel1:
#                 TP_1 = TP_1 + 1
#             elif label[i][j] ==key_pixel1 and pre[i][j] != key_pixel1:
#                 FN_1 = FN_1 + 1
#             elif label[i][j] != key_pixel1 and pre[i][j] == key_pixel1:
#                 FP_1 = FP_1 + 1
#             elif label[i][j] != key_pixel1 and pre[i][j] != key_pixel1:
#                 TN_1 = TN_1 + 1
#             # 目标1的精度
#             if label[i][j]==key_pixel2 and pre[i][j]==key_pixel2:
#                 TP_2 = TP_2 + 1
#             elif label[i][j] ==key_pixel2 and pre[i][j] != key_pixel2:
#                 FN_2 = FN_2 + 1
#             elif label[i][j] != key_pixel2 and pre[i][j] == key_pixel2:
#                 FP_2 = FP_2 + 1
#             elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
#                 TN_2 = TN_2 + 1
#                 #目标3的精度
#             if label[i][j]==key_pixel3 and pre[i][j]==key_pixel3:
#                 TP_3 = TP_3 + 1
#             elif label[i][j] ==key_pixel3 and pre[i][j] != key_pixel3:
#                 FN_3 = FN_3 + 1
#             elif label[i][j] != key_pixel3 and pre[i][j] == key_pixel3:
#                 FP_3 = FP_3 + 1
#             elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
#                 TN_3 = TN_3 + 1
#     #目标1的精度
#     if TP_1+FP_1 !=0:
#         precision1=float(TP_1)/float(TP_1+FP_1)
#     else:
#         precision1=0
#     if TP_1+FN_1 !=0:
#         recall1=float(TP_1)/float(TP_1+FN_1)
#     else:
#         recall1=0
#     # 目标2的精度
#     if TP_2+FP_2 !=0:
#         precision2=float(TP_2)/float(TP_2+FP_2)
#     else:
#         precision2=0
#     if TP_2+FN_2 !=0:
#         recall2=float(TP_2)/float(TP_2+FN_2)
#     else:
#         recall2=0
#     #目标3的准确度
#     if TP_3+FP_3 !=0:
#         precision3=float(TP_3)/float(TP_3+FP_3)
#     else:
#         precision3=0
#     if TP_3+FN_3 !=0:
#         recall3=float(TP_3)/float(TP_3+FN_3)
#     else:
#         recall3=0
#     return precision1,recall1,precision2,recall2,precision3,recall3
def acc_3D(pre,label,key_pixel1=255,key_pixel2=76,key_pixel3=0):
    shape=label.shape
    TP_1=0;FP_1=0;FN_1=0;TN_1=0
    TP_2 = 0; FP_2 = 0; FN_2 = 0; TN_2 = 0
    TP_3 = 0;FP_3 = 0;FN_3 = 0;TN_3 = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            # 目标1的精度
            if label[i][j] == key_pixel1 and pre[i][j] == key_pixel1:
                TP_1 = TP_1 + 1
            elif label[i][j] ==key_pixel1 and pre[i][j] != key_pixel1:
                FN_1 = FN_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] == key_pixel1:
                FP_1 = FP_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] != key_pixel1:
                TN_1 = TN_1 + 1
            # 目标2的精度
            if label[i][j]==key_pixel2 and pre[i][j]==key_pixel2:
                TP_2 = TP_2 + 1
            elif label[i][j] ==key_pixel2 and pre[i][j] != key_pixel2:
                FN_2 = FN_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] == key_pixel2:
                FP_2 = FP_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
                TN_2 = TN_2 + 1
            # 目标3的精度
            if label[i][j]==key_pixel3 and pre[i][j]==key_pixel3:
                TP_3 = TP_3 + 1
            elif label[i][j] ==key_pixel3 and pre[i][j] != key_pixel3:
                FN_3 = FN_3 + 1
            elif label[i][j] != key_pixel3 and pre[i][j] == key_pixel3:
                FP_3 = FP_3 + 1
            elif label[i][j] != key_pixel3 and pre[i][j] != key_pixel3:
                TN_3 = TN_3 + 1
    #目标1的精度
    if TP_1+FP_1 !=0:
        precision1=float(TP_1)/float(TP_1+FP_1)
    else:
        precision1=0
    if TP_1+FN_1 !=0:
        recall1=float(TP_1)/float(TP_1+FN_1)
    else:
        recall1=0
    # 目标2的精度
    if TP_2+FP_2 !=0:
        precision2=float(TP_2)/float(TP_2+FP_2)
    else:
        precision2=0
    if TP_2+FN_2 !=0:
        recall2=float(TP_2)/float(TP_2+FN_2)
    else:
        recall2=0
    # 目标2的精度
    if TP_3+FP_3 !=0:
        precision3=float(TP_3)/float(TP_3+FP_3)
    else:
        precision3=0
    if TP_3+FN_3 !=0:
        recall3=float(TP_3)/float(TP_3+FN_3)
    else:
        recall3 = 0
    return precision1, recall1, precision2, recall2, precision3, recall3
def acc_3D_relx(pre,label,key_pixel1=255,key_pixel2=76,N=3):
    shape=label.shape
    TP_1=0;FP_1=0;FN_1=0;TN_1=0;TP_1_relx=0
    TP_2 = 0;FP_2 = 0;FN_2 = 0;TN_2 = 0;TP_2_relx=0
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            # 目标1的精度
            if IsRight_N(label, i, j, key_pixel1, N) and pre[i][j] ==key_pixel1:
                TP_1_relx = TP_1_relx + 1
            # if label[i][j]==key_pixel1 and pre[i][j]==key_pixel1:
            #     TP_1 = TP_1 + 1
            elif label[i][j] ==key_pixel1 and pre[i][j] != key_pixel1:
                FN_1 = FN_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] == key_pixel1:
                FP_1 = FP_1 + 1
            elif label[i][j] != key_pixel1 and pre[i][j] != key_pixel1:
                TN_1 = TN_1 + 1
            # 目标2的精度
            if IsRight_N(label, i, j, key_pixel2, N) and pre[i][j] ==key_pixel2:
                TP_2_relx = TP_2_relx + 1
            # if label[i][j]==key_pixel2 and pre[i][j]==key_pixel2:
            #     TP_2 = TP_2 + 1
            elif label[i][j] ==key_pixel2 and pre[i][j] != key_pixel2:
                FN_2 = FN_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] == key_pixel2:
                FP_2 = FP_2 + 1
            elif label[i][j] != key_pixel2 and pre[i][j] != key_pixel2:
                TN_2 = TN_2 + 1
    #目标1的精度
    if TP_1_relx+FP_1 !=0:
        pre1_relx=float(TP_1_relx)/float(TP_1_relx+FP_1)
    else:
        pre1_relx=0
    if TP_1_relx+FN_1 !=0:
        rec1_relx=float(TP_1_relx)/float(TP_1_relx+FN_1)
    else:
        rec1_relx=0
    # 目标2的精度
    if TP_2_relx+FP_2 !=0:
        pre2_relx=float(TP_2_relx)/float(TP_2_relx+FP_2)
    else:
        pre2_relx=0
    if TP_2_relx+FN_2 !=0:
        rec2_relx=float(TP_2_relx)/float(TP_2_relx+FN_2)
    else:
        rec2_relx=0
    return pre1_relx,rec1_relx,pre2_relx,rec2_relx
#F1 score
def acc_f1(pre,rec):
    if pre+rec!=0:
        f1=2*pre*rec/(pre+rec)
    else:
        f1=0
    return f1
def acc_iou(pre_label, label):
     #pre_label:预测标签
     #label:已有标签
    prediction=np.array(pre_label,dtype=np.float)
    label = np.array(label, dtype=np.float)
     #计算TP
    multiple=prediction*label
    TP=np.sum(np.sum(multiple/(255*255)))
    sub=prediction-label
    FP = sum(sum(np.where(sub > 0, sub, 0))) / 255
    FN = -sum(sum(np.where(sub < 0, sub, 0))) / 255
    if TP+FP+FN!=0:
         iou=float(TP) / float(TP+FP+FN)
    else:
        iou=0
    return iou



