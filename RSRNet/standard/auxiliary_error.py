'''18/08/22 lpl 17:48
'''
import  numpy as np
import  sys
def Accuracy_2D(prediction,label):
    shape = label.shape
    TP = 0;FP = 0;FN = 0;TN = 0
    precision_road=np.zeros([shape[0],1])
    recall_road=np.zeros([shape[0],1])
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
            precision_road[i]=float(TN)/float(TN+FN)
        if TN+FP !=0:
            recall_road[i]=float(TN)/float(TN+FP)
    return np.mean(precision_road), np.mean(recall_road)
def Accuracy_average_max_min(accuracy_list):
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
def Accuracy_completed_image(accuracy_list,name_list):
    '''This function aims to get the accuracy of one completed image.
    accuracy_list:[:,precision,recall]
    name_list:[:,image_patches_name]
    '''
    accuracy_completed_image=[]
    precision=0;recall=0;count=0
    if len(accuracy_list)!=len(name_list):
        print('These are huge mistakes in test results.')
        return
    cur_name=name_list[0][0]
    cur_name_split=bytes.decode(cur_name).split('_')
    cur_name=cur_name_split[0]
    for i in range (1,len(cur_name_split)-1):
        cur_name=cur_name+'_'+cur_name_split[i]
    (cur_precision, cur_recall) = accuracy_list[0][0]
    if cur_recall >= 0 and cur_recall<=1:
        precision = precision + cur_precision
        recall = recall + cur_recall
        count = count + 1
    elif cur_recall <0:
        accuracy_completed_image.append([cur_precision, cur_recall, cur_name])
    for index_batchs in range(0,len(accuracy_list)):
        for index_batch in range(0, len(accuracy_list[index_batchs])):
            if index_batch ==0 and index_batchs ==0:
                continue
            else:
                next_name=name_list[index_batchs][index_batch]
                next_name_split = bytes.decode(next_name).split('_')
                next_name=next_name_split[0]
                for i in range(1, len(next_name_split)-1):
                    next_name = next_name + '_' + next_name_split[i]
                if cur_name==next_name:
                    (cur_precision,cur_recall)=accuracy_list[index_batchs][index_batch]
                    if cur_recall >=0 and cur_recall<=1 :
                        precision=precision+cur_precision
                        recall=recall+cur_recall
                        count=count+1
                    elif cur_recall<0:
                        accuracy_completed_image.append([cur_precision, cur_recall, cur_name])
                else:
                    if count!=0:
                        precision=precision/count
                        recall=recall/count
                        accuracy_completed_image.append([precision,recall,cur_name])
                    cur_name=next_name
                    precision=0;recall=0;count=0;
    if index_batchs==len(accuracy_list)-1 and cur_name==next_name:
        precision = precision / count
        recall = recall / count
        accuracy_completed_image.append([precision, recall, cur_name])
    if len(accuracy_completed_image)==0:
        print('all patches are one image.')
        precision = precision / count
        recall = recall / count
        accuracy_completed_image.append([precision, recall,cur_name])
    return accuracy_completed_image
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

    TP=0;FP=0;FN=0;TN=0
    for i in range(0,shape[0]*shape[1]*shape[2]):
        if label_result[i]==1 and prediction_result[i]==1:
            TN=TN+1
        elif label_result[i]==0 and prediction_result[i]==0:
            TP=TP+1
        elif label_result[i]==1 and prediction_result[i]==0:
            FP=FP+1
        elif label_result[i]==0 and prediction_result[i]==1:
            FN=FN+1
    if TN+FP !=0:
        precision_road=float(TN)/float(TN+FP)
    else:
        precision_road=0
    if TN+FN !=0:
        recall_road=float(TN)/float(TN+FN)
    else:
        recall_road=0
    #print('TN=%d,FN=%d,FP=%d'%(TN,FN,TP))
    return precision_road,recall_road
def Accuracy_Multi_update(prediction,label):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall for all class'''
    shape=label.shape
    prediction_2=np.reshape(prediction,(shape[0]*shape[1]*shape[2],shape[3]))
    label_2 = np.reshape(label, (shape[0] * shape[1] * shape[2] , shape[3]))
    prediction_result=np.argmax(prediction_2,1)
    label_result=np.argmax(label_2,1)
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

def Accuracy_Multi_Relax(prediction,label,N):
    '''According to the output(prediction) of model and label, we can
    get the precision and recall. A correctly predicted label is that predicted result is matched within N pixels'''
    shape=label.shape
    '''The predicted results that have max probability is viewed as the predicted object.
    if the position of predicted results is 0 , it represents the background , otherwise 1 
    represents road. 
    '''
    prediction=np.array(prediction).reshape(shape)
    prediction_result=np.argmax(prediction,3)
    label_result=np.argmax(label,3)
    TP=0;FP=0;FN=0;TN=0
    TP_FP=0;TP_TN=0
    for batch in range(0,shape[0]):
        label_single=np.reshape(label_result[batch],(shape[1],shape[2]))
        prediction_single=np.reshape(prediction_result[batch],(shape[1],shape[2]))
        for i in range(0,shape[1]):
            for j in  range(0,shape[2]):
                #print('i=%d,j=%d'%(i,j))
                if prediction_single[i][j]==1 and IsRight_N(label_single, i, j, 1, N):
                    TP=TP+1
                # The nummber of 1 in predicted label
                if prediction_single[i][j]==1:
                    TP_FP=TP_FP+1
                # The number of 1 in true label
                if label_single[i][j]==1:
                    TP_TN=TP_TN+1
                # if prediction_single[i][j]==1 and IsRight_N(label_single, i, j, 0, N):
                #     TP=TP+1
                # if prediction_single[i][j]==0 and IsRight_N(label_single, i, j, 1, N):
                #     FP=FP+1
                # if prediction_single[i][j]==1 and IsRight_N(label_single, i, j, 1, N):
                #     FN=FN+1
    if TP_TN !=0:
        precision_road=float(TP)/float(TP_TN)
    else:
        precision_road=0
    if TP_FP !=0:
        recall_road=float(TP)/float(TP_FP)
    else:
        recall_road=0
    #print('TN=%d,FN=%d,FP=%d'%(TN,FN,TP))
    return precision_road,recall_road

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
    '''square shape'''
    return False
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
def convert_labels2C_inverse(multi_label):
    '''Convert the predicted image(Two-channel)to normal label!'''
    shape=multi_label.shape
    if len(shape)!=3 and shape[0]==1:
        multi_label = np.reshape(multi_label, (shape[1] * shape[2], shape[3]))
    elif len(shape)==3:
        multi_label = np.reshape(multi_label, (shape[0] * shape[1], shape[2]))
        multi_label_max = np.argmax(multi_label, 1)
        label = np.zeros(shape[0] * shape[1])
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

