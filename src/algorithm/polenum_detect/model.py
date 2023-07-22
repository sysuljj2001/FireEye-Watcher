import numpy as np
import matplotlib.pyplot as plt  # Graph
from keras.models import Sequential  # ANN 网络结构
from keras.layers import Dense # the layer in  the  ANN
from keras.utils.np_utils import to_categorical
import keras
import keras.utils
from keras import utils as np_utils
import cv2
import os
import matplotlib.image as mpimg
import random

# load model
print('loading model......')  
# 加载model
from keras.models import load_model
save_path = 'num_detect.h5'
model = load_model(save_path)
print('Successful!')  

# 数据读取
path = './my_ref'
my_train_img = []
my_train_label = []
my_test_img = []
my_test_label = []
for i in range(10):
  for filename in os.listdir(path + '/' + str(i)):
    img = cv2.imread(path + '/' + str(i) + '/' + filename, 0)
    img = cv2.resize(img, (28,28))
    my_train_img.append(img)
    my_train_label.append(i)
my_train_img = np.asarray(my_train_img)    
my_train_label = np.asarray(my_train_label)  
print(my_train_img.shape)
print(my_train_label.shape)

# 数据清洗
height = my_train_img.shape[0]
idx = [i for i in range(height)]
random.shuffle(idx)
my_train_img = my_train_img[idx]
my_train_label = my_train_label[idx]
         
# 划分训练集和测试集
# r = 0.25 # 测试比例
my_test_img = my_train_img[:60, :, :]
my_test_label = my_train_label[:60]
my_train_img = my_train_img[65:, :, :]
my_train_label = my_train_label[65:]
print(my_train_img.shape)
print(my_train_label.shape)
print(my_test_img.shape)
print(my_test_label.shape)

# 规范化图片   规范化像素值[0,255]
# 为了使神经网络更好的训练，我们把值设置为[-0.5 , 0.5]
train_img = (my_train_img/255) - 0.5
test_img = (my_test_img/255) - 0.5
# 将 28 * 28 像素图片展成 28 * 28 = 784 维向量
train_img = train_img.reshape((-1,784))
test_img = test_img.reshape((-1,784))
#打印出来
print(train_img.shape)
print(test_img.shape) 

# 评估模型
model.evaluate(
    test_img,
    to_categorical(my_test_label)
)

# 预测图片
predictions = model.predict(test_img[:])
# 输出模型预测 同时和标准值进行比较
print(np.argmax(predictions, axis = 1))
print(my_test_label[:])