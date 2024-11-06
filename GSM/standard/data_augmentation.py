import random
import numpy as np

"""
数据集的数据增强方法
共有八种

"""
def flip_img_map(train_data_, train_label_, sat_w, sat_h):
  _train_data=[]
  _train_label=[]
  
  for i in range(0,len(train_data_)):
    train_data=train_data_[i]
    train_label=train_label_[i]
    a = random.randint(0, 7)
    if a == 0:  # 原图
        train_data_reverse = train_data
        train_label_reverse = train_label

    elif a == 1:
        # 原数据进行翻转:水平
        train_data_reverse = np.flip(train_data, 1)
        train_label_reverse = np.flip(train_label, 1)

    elif a == 2:
        # 原数据进行翻转:垂直
        train_data_reverse = np.flip(train_data, 0)
        train_label_reverse = np.flip(train_label, 0)

    elif a == 3:
        # 数据进行水平翻转后进行rot90
        train_data_res = np.flip(train_data, 1)
        train_label_res = np.flip(train_label, 1)
        train_data_res_resize = np.resize(train_data_res, (sat_w, sat_h, 3))
        train_label_res_resize = np.resize(train_label_res, (sat_w, sat_h, 2))
        train_data_res_resize_rot90 = np.rot90(train_data_res_resize)
        train_label_res_resize_rot90 = np.rot90(train_label_res_resize)
        train_data_reverse = np.resize(train_data_res_resize_rot90, (1, sat_w, sat_h, 3))
        train_label_reverse = np.resize(train_label_res_resize_rot90, (1, sat_w, sat_h, 2))

    elif a == 4:
        # rot 180
        train_data_resize = np.resize(train_data, (sat_w, sat_h, 3))
        train_label_resize = np.resize(train_label, (sat_w, sat_h, 2))
        train_data_resize_rot180 = np.rot90(train_data_resize, 2)
        train_label_resize_rot180 = np.rot90(train_label_resize, 2)
        train_data_reverse = np.resize(train_data_resize_rot180, (1, sat_w, sat_h, 3))
        train_label_reverse = np.resize(train_label_resize_rot180, (1, sat_w, sat_h, 2))

    elif a == 5:
        # rot 90
        train_data_resize = np.resize(train_data, (sat_w, sat_h, 3))
        train_label_resize = np.resize(train_label, (sat_w, sat_h, 2))
        train_data_resize_rot90 = np.rot90(train_data_resize)
        train_label_resize_rot90 = np.rot90(train_label_resize)
        train_data_reverse = np.resize(train_data_resize_rot90, (1, sat_w, sat_h, 3))
        train_label_reverse = np.resize(train_label_resize_rot90, (1, sat_w, sat_h, 2))

    elif a == 6:
        # rot 270
        train_data_resize = np.resize(train_data, (sat_w, sat_h, 3))
        train_label_resize = np.resize(train_label, (sat_w, sat_h, 2))
        train_data_resize_rot180 = np.rot90(train_data_resize, 3)
        train_label_resize_rot180 = np.rot90(train_label_resize, 3)
        train_data_reverse = np.resize(train_data_resize_rot180, (1, sat_w, sat_h, 3))
        train_label_reverse = np.resize(train_label_resize_rot180, (1, sat_w, sat_h, 2))

    elif a == 7:
        # 数据进行垂直翻转后进行rot90
        train_data_res = np.flip(train_data, 0)
        train_label_res = np.flip(train_label, 0)
        train_data_res_resize = np.resize(train_data_res, (sat_w, sat_h, 3))
        train_label_res_resize = np.resize(train_label_res, (sat_w, sat_h, 2))
        train_data_res_resize_rot90 = np.rot90(train_data_res_resize)
        train_label_res_resize_rot90 = np.rot90(train_label_res_resize)
        train_data_reverse = np.resize(train_data_res_resize_rot90, (1, sat_w, sat_h, 3))
        train_label_reverse = np.resize(train_label_res_resize_rot90, (1, sat_w, sat_h, 2)) 
    _train_data.append(train_data)
    _train_label.append(train_label)
  print("数据增强")
  return  _train_data, _train_label