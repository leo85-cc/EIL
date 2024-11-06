'''18/08/22 lpl 17:48
Update1:19/05/31 15:01
（1）添加注释
Basic
'''
import  numpy as np
import  sys
import  time
import  datetime
'''根据日期生成字符串'''
def generate_name_date():
    '''We would generate a name based on the local date'''
    name=datetime.datetime.now().strftime('%Y-%m-%d')
    return name
'''根据时间生成字符串'''
def generate_name_time():
    '''We would generate a name based on the local time'''
    name=time.strftime('%H-%M-%S', time.localtime(time.time()))
    return name
'''Note:此函数已弃用，请勿使用
计算单类别预测目标的精确度，比如，道路或者建筑物，但是不能同时计算两个目标的精确度。标签形式为2D-矩阵，矩阵中的每个元素是一个二维的向量(v1,v2)，v1代表该像素属于背景的的概率。v2代表属于预测目标的概率。'''
def Accuracy_2D(prediction,label):
    '''
    :param prediction: 模型的输出结果
    :param label:实际标签
    :return: precision 和 recall
    '''
    shape = label.shape
    TP = 0;FP = 0;FN = 0;TN = 0
    precision=np.zeros([shape[0],1])
    recall=np.zeros([shape[0],1])
    for i in  range(0,shape[0]):
        for j in range(0,shape[1]):
            if label[i,j] == 1 and prediction[i,j] == 1:
                TN = TN + 1
            elif label[i,j] == 0 and prediction[i,j] == 0:
                TP = TP + 1
            elif label[i,j] == 1 and prediction[i,j] == 0:
                FP = FP + 1
            elif label[i,j] == 0 and prediction[i,j] == 1:
                FN = FN + 1
        if TN+FN !=0:
            precision[i]=float(TN)/float(TN+FN)
        if TN+FP !=0:
            recall[i]=float(TN)/float(TN+FP)
    return np.mean(precision), np.mean(recall)
'''
计算一批影像的平均、最大、最小精确度。此函数主要用于影像拼接以及精确度计算过程中。
'''
def Accuracy_average_max_min(accuracy_list):
    '''
    :param accuracy_list: 精确度列表，该列表中包含了所有影像的name,precision 和recall三个字段
    :return:平均、最大、最小精确度
    '''
    '''This function aims to get three measures: average, man and min.'''
    if len(accuracy_list)==0:
        print('The len of accuracy list is 0.')
        return
    precision,recall,name=accuracy_list[0]
    accu_conut=0;
    max_accuracy=0;min_accuracy=2
    error=0;error_count=0
    sum_precision=0;sum_recall=0
    if recall <0:
        error=error+precision
        error_count=error_count+1
    else:
        accuracy = (precision + recall) / 2
        max_accuracy=accuracy;min_accuracy=accuracy
        max_name = name;min_name = name
        sum_precision=sum_precision+precision
        sum_recall=sum_recall+recall
        accu_conut =accu_conut+1
    for i in range(1,len(accuracy_list)):
        precision,recall,name=accuracy_list[i]
        if recall <0:
            error = error + precision
            error_count = error_count + 1
        else:
            accuracy=(precision+recall)/2
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_name=name
            elif accuracy<min_accuracy:
                min_accuracy=accuracy
                min_name=name
            sum_precision=sum_precision+precision
            sum_recall=sum_recall+recall
            accu_conut = accu_conut + 1
    aver_precision=sum_precision/accu_conut
    aver_recall=sum_recall/accu_conut
    aver_accuracy = (aver_precision+aver_recall) / 2
    if error_count!=0:
        aver_error=float(error)/error_count
    else:
        aver_error=0
    return aver_accuracy,aver_precision,aver_recall,max_accuracy,min_accuracy,max_name,min_name,aver_error
'''计算完整影像的精确度。由于影像过大，因此需要分割后进行逐步预测。此函数根据影像块名，首先拼接影像，然后计算单幅影像的精确度'''
def Accuracy_completed_image(accuracy_list,name_list):
    '''
    :param accuracy_list: 精确度列表，包括含每一个块的precision和recall
    :param name_list: accuracy_list中每一块的名字，其关系是一一对应的
    :return: 一个数组，每个元素为单幅影像的精确度。
    '''
    '''This function aims to get the accuracy of one completed image.
    accuracy_list:[:,precision,recall]
    name_list:[:,image_patches_name]
    '''
    accuracy_completed_image=[]
    precision=0;recall=0;count=0
    error=0;error_count=0
    if len(accuracy_list)!=len(name_list):
        print('These are huge mistakes in test results.')
        return
    cur_name=name_list[0]
    cur_name_split=bytes.decode(cur_name).split('_')
    cur_name=cur_name_split[0]
    #cur_name=cur_name_split[0]+'_'+cur_name_split[1]+'_'+cur_name_split[2]
    (cur_precision, cur_recall) = accuracy_list[0]
    if cur_recall >= 0 and cur_recall<=1:
        precision = precision + cur_precision
        recall = recall + cur_recall
        count = count + 1
    elif cur_recall <0:
        error = error+cur_precision;
        error_count = error_count+1
        #accuracy_completed_image.append([cur_precision, cur_recall, cur_name])
    for index_batchs in range(0,len(accuracy_list)):
        if index_batchs ==0:
            continue
        else:
            next_name=name_list[index_batchs]
            (cur_precision, cur_recall) = accuracy_list[index_batchs]
            next_name_split = bytes.decode(next_name).split('_')
            next_name=next_name_split[0]
            for i in range(1, len(next_name_split)-1):
                next_name = next_name + '_' + next_name_split[i]
            if cur_name==next_name:
                if cur_recall >=0 and cur_recall<=1 :
                    precision=precision+cur_precision
                    recall=recall+cur_recall
                    count=count+1
                elif cur_recall<0:
                    error=error+cur_precision
                    error_count=error_count+1
                    #accuracy_completed_image.append([cur_precision, cur_recall, cur_name])
            else:
                if count!=0:
                    precision=precision/count
                    recall=recall/count
                    accuracy_completed_image.append([precision,recall,cur_name])
                    precision=0;recall=0;count=0
                    if error_count !=0:
                        error = error / error_count
                        accuracy_completed_image.append([error, -1, cur_name])
                        error=0;error_count=0
                cur_name=next_name
                if cur_recall >= 0 and cur_recall <= 1:
                    precision = precision + cur_precision
                    recall = recall + cur_recall
                    count = count + 1
                elif cur_recall < 0:
                    error = error + cur_precision
                    error_count = error_count + 1
    if index_batchs==len(accuracy_list)-1 and cur_name==next_name:
        precision = precision / count
        recall = recall / count
        accuracy_completed_image.append([precision, recall, cur_name])
        if error_count!=0:
            error = error / error_count
            accuracy_completed_image.append([error, -1, cur_name])
    if len(accuracy_completed_image)==0:
        print('all patches are one image.')
        precision = precision / count
        recall = recall / count
        accuracy_completed_image.append([precision, recall,cur_name])
    return accuracy_completed_image
'''计算单类别预测目标的精确度，比如，道路或者建筑物，但是不能同时计算两个目标的精确度。标签形式为2D-矩阵，
矩阵中的每个元素是一个二维的向量(v1,v2)，v1代表该像素属于背景的的概率。v2代表属于预测目标的概率。
Note:accuracy_2D的替代版本。新增加：标签非真，预测非真情况处理。'''
def Accuracy_Multi(prediction,label):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall'''
    shape=label.shape
    if len(shape)==3:
        shape=(1,)+shape
    prediction_2=np.reshape(prediction,(shape[0]*shape[1]*shape[2],shape[3]))
    label_2 = np.reshape(label, (shape[0] * shape[1] * shape[2] , shape[3]))
    prediction_result=np.argmax(prediction_2,1)
    label_result=np.argmax(label_2,1)
    # when test image hasn't the object pixel, we take case as follow.
    if sum(label_result)==0:
        #标签非真，预测非真情况处理方式，2代表100%。
        if sum(prediction_result)==sum(label_result):
            precision = 2
            recall= 2
            return precision, recall
        else:
            # sum(label_result)!=0,we set recall_road to negative as a signal and set precision to error not real precision.
            error=float(sum(prediction_result))/len(prediction_result)
            flag=-1
            return  error,flag
    TP=0;FP=0;FN=0;TN=0
    for i in range(0,shape[0]*shape[1]*shape[2]):
        if label_result[i]==1 and prediction_result[i]==1:
            TP=TP+1
        elif label_result[i]==0 and prediction_result[i]==0:
            TN=TN+1
        elif label_result[i]==1 and prediction_result[i]==0:
            FN=FN+1
        elif label_result[i]==0 and prediction_result[i]==1:
            FP=FP+1
    if TP+FP !=0:
        precision=float(TP)/float(TP+FP)
    else:
        precision=0
    if TP+FN !=0:
        recall=float(TP)/float(TP+FN)
    else:
        recall=0
    #print('TN=%d,FN=%d,FP=%d'%(TN,FN,TP))
    return precision,recall
'''计算两类别预测目标的精确度，比如，道路和建筑物。标签形式为2D-矩阵，
矩阵中的每个元素是一个二维的向量(v1,v2)，v1代表该像素属于背景的的概率。
v2代表属于预测目标的概率。'''
def Accuracy_Multi_update(prediction,label):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall for all class'''
    shape=label.shape
    prediction_2=np.reshape(prediction,(shape[0]*shape[1]*shape[2],shape[3]))
    label_2 = np.reshape(label, (shape[0] * shape[1] * shape[2] , shape[3]))
    prediction_result=np.argmax(prediction_2,1)
    label_result=np.argmax(label_2,1)
    if sum(label_result)==0:
        #标签非真，预测非真情况处理方式，2代表100%。
        if sum(prediction_result)==sum(label_result):
            precision_road = 2
            recall_road = 2
            return precision_road, recall_road
        else:
            '''如果标签全是0,而预测结果有非0，那么我们认为这是预测误差，将recall_road设置为
            -1，作为一个标志'''
            precision_road=float(sum(prediction_result))/len(prediction_result)
            recall_road=-1
            return  precision_road,recall_road
    TP=0;FP=0;FN=0;TN=0
    for i in range(0,shape[0]*shape[1]*shape[2]):
        if label_result[i]==1 and prediction_result[i]==1:
            TP=TP+1
        elif label_result[i]==0 and prediction_result[i]==0:
            TN=TN+1
        elif label_result[i]==1 and prediction_result[i]==0:
            FN=FN+1
        elif label_result[i]==0 and prediction_result[i]==1:
            FP=FP+1
    # the precision of class 1
    if TN+FN !=0:
        precision1=float(TN)/float(TN+FN)
    else:
        precision1=0
    # the recall of class 1
    if TN+FP !=0:
        recall1=float(TN)/float(TN+FP)
    else:
        recall1=0
    # the precision of class 2
    if TP+FP!=0:
        precision2=float(TP)/float(TP+FP)
    else:
        precision2=0;
    # the recall of class 2
    if TP+FN!=0:
        recall2=float(TP)/float(TP+FN)
    else:
        recall2=0
    #print('TN=%d,FN=%d,FP=%d'%(TN,FN,TP))
    return precision1,recall1,precision2,recall2
'''松散式精度计算。该函数基于Accuracy_Multi（）函数，只可用于单目标的精度计算'''
def Accuracy_Multi_Relax(prediction,label,N):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall. A correctly predicted label is that predicted result is matched
    within N pixels'''
    shape=label.shape
    prediction=np.array(prediction).reshape(shape)
    prediction_result=np.argmax(prediction,3)
    label_result=np.argmax(label,3)
    # when test image hasn't the object pixel, we take case as follow.
    if sum(label_result.flatten()) == 0:
        # 标签非真，预测非真情况处理方式，2代表100%。
        if sum(prediction_result.flatten()) == sum(label_result.flatten()):
            precision = 2
            recall= 2
            return precision, recall
        else:
            '''如果标签全是0,而预测结果有非0，那么我们认为这是预测误差，将recall_road设置为
            -1，作为一个标志'''
            error = float(sum(prediction_result.flatten())) / (shape[0]*shape[1]*shape[2])
            error_flag = -1
            return error, error_flag
    TP=0;FP=0;FN=0;TN=0;TP_relax=0
    for batch in range(0,shape[0]):
        label_single=np.reshape(label_result[batch],(shape[1],shape[2]))
        prediction_single=np.reshape(prediction_result[batch],(shape[1],shape[2]))
        for i in range(0,shape[1]):
            for j in  range(0,shape[2]):
                if prediction_single[i][j]==1 and IsRight_N(label_single, i, j, 1, N):
                    TP_relax=TP_relax+1
                if label_single[i][j] == 1 and prediction_single[i][j] == 1:
                    TP = TP + 1
                elif label_single[i][j] == 0 and prediction_single[i][j] == 0:
                    TN = TN + 1
                elif label_single[i][j] == 1 and prediction_single[i][j] == 0:
                    FN = FN + 1
                elif label_single[i][j] == 0 and prediction_single[i][j] == 1:
                    FP = FP + 1
    if TP_relax+FN !=0:
        recall=float(TP_relax)/float(TP_relax+FN)
    else:
        recall=0
    if TP_relax+FP !=0:
        precision=float(TP_relax)/float(TP_relax+FP)
    else:
        precision=0
    #print('TN=%d,FN=%d,FP=%d'%(TN,FN,TP))
    return precision,recall
'''判断预测像素的临近范围内是否有相应对的标签。邻域大小为该像素的上、小、左、右个N像素'''
def IsRight_N(label,pos_row,pos_column,prediction,N):
    label_shape=label.shape
    start_row=0 if pos_row-N<0 else pos_row -N
    end_row=label_shape[0]-1 if pos_row+N>=label_shape[0] else pos_row+N
    start_column= 0 if pos_column-N<0 else pos_column-N;
    end_column=label_shape[1]-1 if pos_column+N>=label_shape[1] else pos_column+N
    '''corss shape'''
    for i in range(start_row,end_row+1):
        if label[i][pos_column]==prediction:
            return True
    for j in range(start_column,end_column+1):
        if label[pos_row][j]==prediction:
            return True
    return False
'''已弃用，请勿使用'''
def Accuracy(prediction,label):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall'''
    shape=prediction.shape
    prediction_2=np.reshape(prediction,shape[0]*shape[1]*shape[2],shape[3])
    label_2 = np.reshape(label, shape[0] * shape[1] * shape[2] , shape[3])
    TP=0;FP=0;FN=0;TN=0
    for i in range(0,shape[0]*shape[1]*shape[2]):
        if label_2[i]==1 and prediction_2[i]==1:
            TN=TN+1
        elif label_2[i]==0 and prediction_2[i]==0:
            TP=TP+1
        elif label_2[i]==1 and prediction_2[i]==0:
            FP=FP+1
        elif label_2[i]==0 and prediction_2[i]==1:
            FN=FN+1
    if TN+FN !=0:
        precision_road=float(TN)/float(TN+FN)
    else:
        precision_road=0
    if TN+FP !=0:
        recall_road=float(TN)/float(TN+FP)
    else:
        recall_road=0
    return precision_road,recall_road
'''已弃用，请勿使用'''
def Multi_channel_labels2_labels(multi_channel_labels):
    '''recover the label'''
    shape = multi_channel_labels.shape
    label = np.zeros([shape[0], shape[1], 1], dtype=np.float32)
    for horizontal_i in range(0,shape[0]):
        for vertical_i in range(0, shape[1]):
            if multi_channel_labels[horizontal_i][vertical_i][0]==1:
                label[horizontal_i][vertical_i]=0
                break
            elif multi_channel_labels[horizontal_i][vertical_i][1]==1:
                label[horizontal_i][vertical_i] = 255
    return label
'''
将预测的predicted-label-map转化为相应的真实label-map，该label-map中的元素为像素值而不是概率，例如,0代表黑色，255代表白色。
'''
def convert_labels2C_inverse(multi_label):
    '''Convert the predicted image(Two-channel)to normal label!'''
    shape=multi_label.shape

    if len(shape)!=3 and shape[0]==1:
        multi_label = np.reshape(multi_label, (shape[1] * shape[2], shape[3]))
    elif len(shape)==3:
        multi_label = np.reshape(multi_label, (shape[0] * shape[1], shape[2]))
        multi_label_max = np.argmax(multi_label, 1)
        label = np.zeros(shape[0] * shape[1],dtype=np.uint8)
        for i in range(0, shape[0] * shape[1]):
            if multi_label_max[i] == 0:
                label[i] = 0
            elif multi_label_max[i] == 1:
                label[i] = 255
            else:
                print('The ', i, '-th pixel of multi-label is illegal.')
    else:
        print('The format of multi-label is error!')
        sys.exit()
    return np.reshape(label,(shape[0],shape[1]))
'''将label-map中的每个像素转化为向量。例如：label-map:(1,0;0,1),转化后为((0,1),(1,0);(1,0),(0,1))。其中向量中的第一元素代表是否为背景，第二个元素代表是否为目标。1表示真，0表示假。
Note:函数只适用于单类目标预测'''
def Convert_Labels(source_labels):
    '''For the train data (Mnih's data)that label-patch is a 2-D matrix,this method
    convert source labels to 3-D matrix.In this matrix, each of label pixel have
    two channel. First one is represent backgroud channel and second one is road channel.
    such as:source labels is (1,0;0,1),dst label is ((0,1),(1,0);(1,0),(0,1))
    '''
    shape=source_labels.shape
    if len(shape)==2:
        dst_labels = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
        for horizontal_i in range(0, shape[0]):
            for vertical_i in range(0, shape[1]):
                if source_labels[horizontal_i, vertical_i] == 0:
                    dst_labels[horizontal_i, vertical_i, 0] = 1
                else:
                    dst_labels[horizontal_i, vertical_i, 1] = 1
    else:
        dst_labels=np.zeros([shape[0],shape[1],shape[2],2],dtype=np.float32)
        for batchsize_i in range(0,shape[0]):
            for horizontal_i in range(0,shape[1]):
                for vertical_i in range (0,shape[2]):
                    if source_labels[batchsize_i,horizontal_i,vertical_i]==0:
                        dst_labels[batchsize_i,horizontal_i,vertical_i,0]=1
                    else:
                        dst_labels[batchsize_i, horizontal_i, vertical_i, 1] = 1
    return dst_labels
'''将label_map中的每个像素转化为3D向量,改函数主要用于三分类。例如：label-map:(1,2;0,1),
转化后为((0,10),(0,0,1);(1,0,0),(0,1,0))。其中向量中的第一元素代表是否为背景，第二个元素代表是否为目标1,
第二个元素代表是否为目标2。1表示真，0表示假。
Note:函数只适用于单类目标预测'''
#未完成
def convert_3D(label_map,pixel_value1,pixel_value2):
    shape=source_labels.shape
    if len(shape)==2:
        dst_labels = np.zeros([shape[0], shape[1], 2], dtype=np.float32)
        for horizontal_i in range(0, shape[0]):
            for vertical_i in range(0, shape[1]):
                if source_labels[horizontal_i, vertical_i] == 0:
                    dst_labels[horizontal_i, vertical_i, 0] = 1
                else:
                    dst_labels[horizontal_i, vertical_i, 1] = 1
    else:
        dst_labels=np.zeros([shape[0],shape[1],shape[2],2],dtype=np.float32)
        for batchsize_i in range(0,shape[0]):
            for horizontal_i in range(0,shape[1]):
                for vertical_i in range (0,shape[2]):
                    if source_labels[batchsize_i,horizontal_i,vertical_i]==0:
                        dst_labels[batchsize_i,horizontal_i,vertical_i,0]=1
                    else:
                        dst_labels[batchsize_i, horizontal_i, vertical_i, 1] = 1
    return dst_labels
'''已弃用，请勿使用'''
def Convert_Vector(data):
    '''For the train data (Mnih's data)that label-patch is a 2-D matrix,this method
        convert source labels to vector.
        such as:source labels is (1,0;0,1),dst label is (1,0,0,1)
        '''
    data_shape=data.shape
    dst_data=np.zeros([data_shape[0],data_shape[1]*data_shape[2]])
    for batch_i in range(0,data_shape[0]):
        data_batch=data[batch_i]
        data_vector=data_batch.reshape(1,data_shape[1]*data_shape[2])
        dst_data[batch_i,:]=data_vector
    return  dst_data
'''测试程序接口，请勿使用'''
if __name__ =='__main__':
    generate_name_date()
    generate_name_time()
