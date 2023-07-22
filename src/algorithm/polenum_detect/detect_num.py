import pytesseract
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import scipy
from keras.models import Sequential  # ANN 网络结构
from keras.layers import Dense # the layer in  the  ANN
from keras.utils.np_utils import to_categorical
import keras
import keras.utils
from keras import utils as np_utils
import matplotlib.image as mpimg
import random

#模版匹配
# 准备模板(template[0-9]为数字模板；)
template = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',
            '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']

# 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

# 获得数字模板列表（匹配车牌后面的字符）
def get_num_words_list():
    num_words_list = []
    for i in range(10):
        word = read_directory('./my_ref/'+ template[i])
        num_words_list.append(word)
    return num_words_list
num_words_list = get_num_words_list()


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    #将模板进行格式转换
    template_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    #模板图像阈值化处理——获得黑白图
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
#     height, width = template_img.shape
#     image_ = image.copy()
#     image_ = cv2.resize(image_, (width, height))
    image_ = image.copy()
    #获得待检测图片的尺寸
    height, width = image_.shape
    # 将模板resize至与图像一样大小
    template_img = cv2.resize(template_img, (width, height))
    # 模板匹配，返回匹配得分
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]

# 对分割得到的字符逐一匹配
def template_matching(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        best_score = []
        for num_word_list in num_words_list:
            score = []
            for num_word in num_word_list:
                result = template_score(num_word,word_image)
                score.append(result)
            best_score.append(max(score))
        i = best_score.index(max(best_score))
        # print(template[i])
        r = template[i]
        results.append(r)
        continue
    return results

# plt显示彩色图片
def plt_show0(img):
#cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    cv2.imshow("title", img)
    cv2.waitKey(0)
    
# plt显示灰度图片
def plt_show(img):
    cv2.imshow("title", img)
    cv2.waitKey(0)

# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    # 因为这个函数除了旋转外还有平移量，因此需要返回平移量M[0, 2], M[1, 2]以供后续图像处理
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255)), M[0, 2], M[1, 2]
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))

# 绕pointx,pointy顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
  angle = angle * np.pi / 180
  valuex = np.array(valuex)
  valuey = np.array(valuey)
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return sRotatex,sRotatey

# step代表第几次调用该函数
def Contours(image_crop, step):
    # 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作
    image = image_crop
    ret, image = cv2.threshold(image_crop, 127, 255, cv2.THRESH_BINARY_INV)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 40))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
    
    # 显示灰度图像
    # cv2.imshow("2", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    # 腐蚀（erode）和膨胀（dilate）
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    if step: # step = 1
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        # x方向进行操作
        image = cv2.dilate(image, kernelX)
        image = cv2.erode(image, kernelx)
        # y方向操作
        image = cv2.erode(image, kernelY)
    else: # step = 0
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        # x方向进行操作
        image = cv2.dilate(image, kernelX)
        image = cv2.erode(image, kernelx)
        # y方向操作
        image = cv2.erode(image, kernelY)
    # 中值滤波（去噪）
    image = cv2.medianBlur(image, 21)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # 显示灰度图像
    # cv2.imshow("3", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    # 获得轮廓
    pt = 0 # 是否检测到轮廓的标记
    im, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xx = yy = w = h = 0
    if contours:
        for item in contours:
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            # 根据轮廓的形状特点，确定车牌的轮廓位置并截取图像
            if (height > (weight * 2)) and (height < (weight * 12)) and height*weight > 3000:
                image = image_crop[y:y + height, x:x + weight]
                h = height
                w = weight
                xx = x
                yy = y
                pt = 1
                break
    return xx, yy, h, w, pt


def detect(origin_image, filenum, model):
    # 读取待检测图片
    # origin_image = cv2.imread('168.png')
    # 缩放
    size = 480
    # 获取原始图像宽高。
    height, width = origin_image.shape[0], origin_image.shape[1]
    # 等比例缩放尺度。
    scale = height/size
    # 获得相应等比例的图像宽度。
    width_size = int(width/scale)
    # resize
    image_resize = cv2.resize(origin_image, (width_size, size))
    # 旋转
    # 复制一张图片，在复制图上进行图像操作，保留原图
    image = image_resize.copy()
    # 图像去噪灰度处理
    gray_image = gray_guss(image)
    # x方向上的边缘检测（增强边缘信息）
    #Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    #absX = cv2.convertScaleAbs(Sobel_x)
    #image = absX
    image = gray_image

    # LSD
    lsd = cv2.createLineSegmentDetector(0, _scale=1)
    dlines = lsd.detect(image)
    length = []
    cord = []
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        l = np.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))
        length.append(l)
        cord.append([x0, y0, x1, y1])

    # 获取最长的四条直线，并计算斜率，旋转图像
    gray_image_lines = gray_image
    sorted_id = sorted(range(len(length)), key=lambda k: length[k], reverse=True)
    k = [] # 斜率
    center = [] # 中点值
    for i in range(4):
        cv2.line(gray_image_lines, (cord[sorted_id[i]][0], cord[sorted_id[i]][1]), \
            (cord[sorted_id[i]][2], cord[sorted_id[i]][3]), 0, 3, cv2.LINE_AA)
        rate = (cord[sorted_id[i]][0] - cord[sorted_id[i]][2]) / (cord[sorted_id[i]][1] - cord[sorted_id[i]][3] + 0.0001)
        x_cen = (cord[sorted_id[i]][0] + cord[sorted_id[i]][2]) / 2
        y_cen = (cord[sorted_id[i]][1] + cord[sorted_id[i]][3]) / 2
        k.append(rate)
        center.append([x_cen, y_cen])
    k_avg = sum(k) / len(k)
    theta = np.arctan(k_avg) / np.pi * 180
    gray_image, t1, t2 = rotate_bound_white_bg(gray_image, theta)
    gray_image_lines, tl1, tl2 = rotate_bound_white_bg(gray_image_lines, theta) # 这个是拿来画图的
    
    # 裁剪杆号区域
    y, x = gray_image.shape
    l_cenx = [] # 左，右center
    r_cenx = []
    for i in range(len(center)):
        center[i] = Srotate(-theta, center[i][0], center[i][1], x/2, y/2) # 将center点旋转
        center[i] = [center[i][0]+t1/2, center[i][1]+t2/2] # 补充平移余量
        cv2.circle(gray_image_lines, (int(round(center[i][0])), int(round(center[i][1]))), 5, 0, 5)
        if center[i][0] < x / 2: l_cenx.append(center[i][0])
        else: r_cenx.append(center[i][0])

    # cv2.imshow("0", gray_image_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    # image_adaptive = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,155,0)
    # ret, image_adaptive = cv2.threshold(gray_image, 87, 255, cv2.THRESH_BINARY)
    if len(l_cenx) and len(r_cenx):
        x_l = int(round(sum(l_cenx) / len(l_cenx))) # 左边线中点值
        x_r = int(round(sum(r_cenx) / len(r_cenx)))
        if x_l < x_r - 20:
            image_crop = gray_image[:, x_l+10:x_r-10] # 10是余量
            tal = np.sum(image_crop) / (image_crop.shape[0] * image_crop.shape[1]) # 平均亮度
            # 统计直方图
            hist = cv2.calcHist([image_crop], [0], None, [256], [0, 255])
            hist = hist.transpose()[0]
            hist_smooth = scipy.signal.savgol_filter(hist,11,3) 
            peak_id = np.argmax(hist_smooth)
            thred = peak_id / 3 # 取峰值一半
            # plt.plot(hist_smooth, color="r")
            # plt.show()
            # print(thred) 
            ret, image_crop = cv2.threshold(image_crop, thred, 255, cv2.THRESH_BINARY)
            # cv2.imshow("1", image_crop)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

            # 形态学操作
            xx, yy, h, w, pt = Contours(image_crop, 0)

            # 图像阈值化操作——获得二值化图
            if pt:   
                if xx-15 < 0 or yy-20 < 0:
                    image = image_crop[yy:yy + h, xx:xx + w]
                else:
                    image = image_crop[yy - 20:yy + h + 20, xx - 15:xx + w + 15] # 留足余量
                ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                image = cv2.erode(image, kernel)
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # image = cv2.dilate(image, kernel)
                # image = cv2.GaussianBlur(image, (3, 3), 0)
                # cv2.imshow("5", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows() 

                # 手动寻找上下左右边界
                up = 0
                down = 0
                left = 0
                right = 0
                # 先左右
                for i in range(image.shape[1]-1, -1, -1): # 倒序
                    col = image[:, i].tolist()
                    if col.count(255) == 0: 
                        right = i
                        break
                for i in range(image.shape[1]):
                    col = image[:, i].tolist()
                    if col.count(255) == 0: 
                        left = i
                        break
                if right > left:
                    image = image[:, left:right]
                else: return [-1]

                # 再上下
                for i in range(image.shape[0]):
                    row = image[i, :].tolist()
                    if row.count(255): 
                        up = i
                        break
                for i in range(image.shape[0]-1, -1, -1): # 倒序
                    row = image[i, :].tolist()
                    if row.count(255): 
                        down = i
                        break
                if up < down:
                    if up > 5 and down + 5 < image.shape[0]:
                        image = image[up-5:down+5, :]
                    else:
                        image = image[up:down, :]
                else: return [-1]

                # 再次形态学
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                image = cv2.dilate(image, kernel)
                ret, imageC = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
                xx, yy, h, w, pt = Contours(imageC, 1)
                if pt:   
                    image = image[yy:yy + h, :]
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # image = cv2.dilate(image, kernel)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # image = cv2.dilate(image, kernel)
                    image = cv2.GaussianBlur(image, (3, 3), 0)
                    # cv2.imshow("6", image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows() 

                    # 字符分割, 获得每个数字的轮廓
                    word_images = [] # 保存每个数字图片
                    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
                    imageNum = cv2.dilate(image, kernelX)
                    im, contours, hierarchy = cv2.findContours(imageNum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    xx = yy = w = h = 0
                    # 如果能分出四个块
                    if len(contours) == 4:
                        cnt = 0
                        for item in contours[::-1]:
                            cnt += 1
                            if cnt == 1: continue
                            rect = cv2.boundingRect(item)
                            x = rect[0]
                            y = rect[1]
                            weight = rect[2]
                            height = rect[3]
                            # 根据轮廓的形状特点，确定车牌的轮廓位置并截取图像              
                            num = image[y:y + height, x:x + weight]
                            word_images.append(num)
                            # cv2.imwrite(str(filenum) + '_' + str(cnt) + '.png', num)
                            # cv2.imshow("7", num)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows() 
                    else: # 如不能分出4个块,那么就直接平分成四段
                        h = int(image.shape[0] / 4)
                        for i in range(3):
                            num = image[h*(i+1):h*(i+2), :]
                            word_images.append(num)

                else: return [-1]

                # 方法一: 模板匹配
                # word_images_ = word_images.copy()
                # # 调用函数获得结果
                # result = template_matching(word_images_[:])
                # print(result)
                # "".join(result)函数将列表转换为拼接好的字符串，方便结果显示
                # print( "".join(result))
                
                # 方法二: tesseract字符识别
                # config = r'--oem 3 --psm 6 outputbase digits'
                # code = pytesseract.image_to_string(image, config = config)
                # print(code)
                # print('done')  

                # 方法三: 神经网络识别
                word_images_ = word_images.copy()
                for i in range(3): # 转换成28*28
                    word_images_[i] = cv2.resize(word_images_[i], (28,28))
                word_images_ = np.asarray(word_images_)
                # 规范化图片   规范化像素值[0,255]
                word_images_ = (word_images_/255) - 0.5
                # 将 28 * 28 像素图片展成 28 * 28 = 784 维向量
                word_images_ = word_images_.reshape((-1,784))
                predictions = model.predict(word_images_[:])
                # 输出模型预测 同时和标准值进行比较
                result = np.argmax(predictions, axis = 1)
                # print(result)

                return result
            else: return [-1]
        else: return [-1]
    else: return [-1]


# main
def main(path):
    # load model
    print('loading model......')  
    from keras.models import load_model
    save_path = 'num_detect.h5'
    model = load_model(save_path)
    print('Successful!')  

    # # load img
    # file_path = 'num_clear'
    # # 按照数字从小到大读取
    # sort_num_list = []
    # for filename in os.listdir(file_path):
    #     sort_num_list.append(int(filename.split('.png')[0]))
    #     sort_num_list.sort() #然后再重新排序
    # for filenum in sort_num_list:
    #     # if filenum < 7130: continue
    #     file = str(filenum).rjust(5,'0') + '.png'
    #     print(file)
    #     img_path = file_path+'/'+ file
    img = cv2.imread(path)

    # detect
    num = detect(img, 1, model)
    print(num)
    





         
	
