# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:58:26 2017

@author: rd0348
"""

import cv2 as cv  
import numpy as np  
from matplotlib import pyplot as plt
 
img = cv.imread('fft.jpg', 0) # 直接读取的时候就转为灰度图
f = np.fft.fft2(img) # 快速傅里叶变换算法得到频率分布  
fshift = np.fft.fftshift(f) # 默认结果中心点位置是在左上角，转移到中间位置
 
fimg = np.log(np.abs(fshift)) # fft 结果是复数，求绝对值结果才是振幅
 
# 展示结果
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')  
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')  
plt.show()