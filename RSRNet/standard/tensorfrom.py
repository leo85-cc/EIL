import cv2
import numpy as np


'''
interface: 
rotate_image(sat,map,dst_sat_width,dst_sat_width,dst_map_width,dst_map_high,
            rotate_angle, crop)
the meaning of each parameter:
sat: The sat is a image digit matrix (sat is image as your do in the program)
map: map is the label image corresponding sat

dst_sat_width & dst_sat_width: These two parameters represent the size of croping 
image. Usually, the width and high is equal.

dst_map_width & dst_map_width: These two parameters represent the size of croping 
map. Usually, the width and high is equal. But it is notice is that the size of map
is not same with the size of sat in most of case.
rotate_angle: The angle we need to do for image
crop: when crop is true, it represents that we need to crop the image to elimate 
the invalid edge. Otherwise ,we do not do that.
'''

def rotate_image(sat, map, dst_sat_width, dst_sat_hight, dst_map_width, dst_map_hight, rotate_angle):

    def rotate(img, angle):
        # 进行旋转操作
        w, h, depth = img.shape  # w ：宽  h ：高
        # 旋转角度的周期是360°
        angle %= 360  # angle = angle % 360  取模
        # 计算仿射变换矩阵
        M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 得到旋转后的图像
        img_rotated = cv2.warpAffine(img, M_rotation, (w, h))
        return img_rotated

    rotateSat = rotate(sat, rotate_angle)
    rotateMap = rotate(map, rotate_angle)


    def crop(img, width, hight, angle):
        w, h = img.shape[:2]
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        # 裁剪图片，从左上角（x0,y0）裁剪指定大小的图片
        cropImg = img[y0:y0+hight, x0:width+x0]
        return cropImg

    cropSat = crop(rotateSat, dst_sat_width, dst_sat_hight, rotate_angle)
    cropMap = crop(rotateMap, dst_map_width, dst_map_hight, rotate_angle)

    return cropSat, cropMap
if  __name__=='__main__':
    #裁剪出92*92大小的图片，最大需要131*131大小的原图片
    #裁剪出24*24大小的图片，最大需要31*31大小的原图片
    satPath = "E:/Test/image/3.tiff"
    mapPath = "E:/Test/image/4.tif"
    img1 = cv2.imread(satPath)
    img2 = cv2.imread(mapPath)
    image_rotated, map_rotated = rotate_image(img1, img2, 1060, 1060, 1080, 1080, 45)
    cv2.imwrite("E:/Test/image/Test11.tiff", image_rotated)
    cv2.imwrite("E:/Test/image/Test22.tif", map_rotated)
    print('sucessful')

