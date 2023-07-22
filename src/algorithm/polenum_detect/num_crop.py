# coding=utf-8

import requests
import os
import numpy as np
import sys
import detect_num
import cv2

if __name__ == '__main__':
    # load model
    print('loading model......')  
    from keras.models import load_model
    save_path = 'num_detect.h5'
    model = load_model(save_path)
    print('Successful!')  
    res_path = "detect_res" 

    # 按照数字从小到大读取
    sort_num_list = []
    for filename in os.listdir(res_path):
        sort_num_list.append(int(filename.split('.jpg.txt')[0]))
        sort_num_list.sort() #然后再重新排序

    for filenum in sort_num_list:
        # if filename < 1272: continue
        classes = []
        pole_boxes = []
        num_boxes = []
        pole_num = []
        filename = str(filenum).rjust(5,'0') + '.jpg.txt'
        with open(res_path+'/'+ filename ,'r',encoding='utf-8') as f:
            res = f.read() # .read()能直接读成string
        print(filename[:-8])

        ## 提取classes
        for i in range(len(res)):
            if res[i:i+4] == 'pole': classes.append('pole')
            if res[i:i+3] == 'num': classes.append('num')

        ## 提取框脚点坐标
        sta = res.find('detection_boxes')
        end = res.find('detection_scores')
        num = res[sta+18 : end-3]
        num = num.replace('[', ' ')
        num = num.replace(',', ' ')
        num = num.replace(']', ' ')
        num = num.split()
        # 分类载入
        for i in range(len(classes)):
            if (classes[i] == 'pole'): pole_boxes.append([float(num[4*i]), float(num[4*i+1]), float(num[4*i+2]), float(num[4*i+3])])
            else: num_boxes.append([float(num[4*i]), float(num[4*i+1]), float(num[4*i+2]), float(num[4*i+3])])

        ## 判断杆标号是否在杆框内，并检测杆号
        for i in range(len(pole_boxes)):
            p1_x = pole_boxes[i][0]
            p1_y = pole_boxes[i][1]
            p2_x = pole_boxes[i][2]
            p2_y = pole_boxes[i][3]
            for j in range(len(num_boxes)):
                n1_x = num_boxes[j][0]
                n1_y = num_boxes[j][1]
                n2_x = num_boxes[j][2]
                n2_y = num_boxes[j][3]
                
                if(n1_x>p1_x and n1_y>p1_y and n2_x<p2_x and n2_y<p2_y):
                    ## 功能一: 裁剪图片 (注，；这里有一个bug，即每帧最多只保存一个杆号区域，不想改了)
                    # k = 20
                    # img = cv2.imread('0223_Img' + '/' + filename[:-4])
                    # img = img[int(n1_x-k):int(n2_x+k), int(n1_y-k):int(n2_y+k)] # 截取杆号区域
                    # cv2.imwrite('num_crop'+ '/' + filename[:-8] + '.png', img)
                    # # cv2.imshow("crop_img", img)
                    # # cv2.waitKey(0)

                    ## 功能二: 检测杆号并保存
                    img = cv2.imread('num_clear' + '/' + filename[:-8] + '.png')
                    num = detect_num.detect(img, filenum, model)
                    if num[0]:
                        num = [eval(''.join([str(i) for i in num]))]
                    print(num)
                    res = [p1_x, p1_y, p2_x, p2_y, num[0]]
                    np.set_printoptions(suppress=True)
                    np.set_printoptions(precision=0)   #设精度
                    np.savetxt('detect_num_res/' + filename[:-8] + '.txt', res, fmt='%.00f')

                
            







