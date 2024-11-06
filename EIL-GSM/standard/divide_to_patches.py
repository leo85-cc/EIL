'''lpl 18/08/07 10:35
The divide_to_patches is to get patches accodring to the paprmeters stride,
sat_size, map_size, sat_im and map_im.
The return is the patches of sat_im and map_im
Basic
'''
import numpy as np
import cv2 as cv
'''对影像进行切分，不进行占比控制'''
def divide_to_patches(stride,sat_size,map_size,sat_im,map_im):
    sat_img=sat_im
    map_img=map_im
    sat_patches=[]
    map_patches=[]
    image_count=0
    y=0
    while y <=sat_img.shape[0] - sat_size:
        x = 0
        while x<=sat_img.shape[1] - sat_size:
            if y+sat_size>sat_img.shape[0]:
                y=sat_img.shape[0]-sat_size
            if x+sat_size>sat_img.shape[1]:
                x=sat_img.shape[1]-sat_size
            sat_patch=sat_img[x:x+sat_size,y:y+sat_size]
            sat_size_half=int(sat_size / 2)
            map_size_half=int(map_size / 2)
            map_patch=map_img[
                      x + sat_size_half - map_size_half:x + sat_size_half + map_size_half,
                      y + sat_size_half - map_size_half:y + sat_size_half + map_size_half]
            #print ('sat_patch: ', sat_patch.shape, ' map_patch: ', map_patch.shape)
            #if sat_patch.shape[1]==256:
            sat_patches.append(sat_patch)
            map_patches.append(map_patch)
            image_count = image_count + 1
            x = x + stride
        y=y+stride
    print('finished')
    return sat_patches, map_patches
'''对影像进行切割，并进行占比(ratio)控制
Note:占比关键像素值为keyPixel,该函数仅支持单目标占比控制'''
def divide_to_patches_ratio(stride,sat_size,map_size,sat_im,map_im,ratio,keyPixel):
    sat_img=sat_im
    map_img=map_im
    sat_patches=[]
    map_patches=[]
    image_count=0
    y=0
    while y <=sat_img.shape[0] - sat_size:
        x = 0
        while x<=sat_img.shape[1] - sat_size:
            if y+sat_size>sat_img.shape[0]:
                y=sat_img.shape[0]-sat_size
            if x+sat_size>sat_img.shape[1]:
                x=sat_img.shape[1]-sat_size
            sat_patch=sat_img[x:x+sat_size,y:y+sat_size]
            sat_size_half=int(sat_size / 2)
            map_size_half=int(map_size / 2)
            map_patch=map_img[
                      x + sat_size_half - map_size_half:x + sat_size_half + map_size_half,
                      y + sat_size_half - map_size_half:y + sat_size_half + map_size_half]
            sum_map_values = 0
            for yy in  range(0,map_size):
                for xx in range(0,map_size):
                    if map_patch[xx,yy]==keyPixel:
                        sum_map_values=sum_map_values+1
            if sum_map_values >ratio*map_size*map_size:
                sat_patches.append(sat_patch)
                map_patches.append(map_patch)
                image_count = image_count + 1
            x = x + stride
        y=y+stride
    print('finished')
    return sat_patches, map_patches
'''测试函数，请勿使用'''
if __name__ =='__main__':
    print("divide_to_patches")
    sat_fn = '10078675_15.tiff'
    #map_fn = '/root/lpl/ssai-cnn/data/mass_roads/train/map/17578855_15.tif'
    map_fn = '10078675_15.tif'
    sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
    map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
    divide_to_patches(46, 92, 92, sat_im, map_im,255)
