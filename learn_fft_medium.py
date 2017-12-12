# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:58:46 2017

@author: rd0348
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 支持中文显示
from matplotlib.font_manager import FontProperties  
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 
    
img = cv2.imread('panda.jpg',0) #直接读为灰度图像
f = np.fft.fft2(img) #快速傅里叶变换算法得到频率分布
fshift = np.fft.fftshift(f) #[移频]默认结果中心点位置是在左上角，转移到中间位置
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift)) #fft 结果是复数，求绝对值结果才是振幅
plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original-傅里叶变换后',fontproperties=font_set)
plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center-变换后再移频的',fontproperties=font_set)
plt.show()

ph_f = np.angle(f) #求相位角度
ph_fshift = np.angle(fshift) #求移频后的相位角度

plt.subplot(121),plt.imshow(ph_f,'gray'),plt.title('原始相位图',fontproperties=font_set)
plt.subplot(122),plt.imshow(ph_fshift,'gray'),plt.title('移频后的相位图',fontproperties=font_set)
plt.show()

# 逆变换
f1shift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f1shift)
#出来的是复数，无法显示
img_back = np.abs(img_back)
plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('img back',fontproperties=font_set)
plt.show()

#---------------------------------------------
# 逆变换--取绝对值就是振幅
f1shift = np.fft.ifftshift(np.abs(fshift))
img_back = np.fft.ifft2(f1shift)
#出来的是复数，无法显示
img_back = np.abs(img_back)
#调整大小范围便于显示
img_back = (img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))
plt.subplot(222),plt.imshow(img_back,'gray'),plt.title('仅仅包含振幅',fontproperties=font_set)
plt.xticks([]),plt.yticks([])
plt.show()
#---------------------------------------------
# 逆变换--取相位
f2shift = np.fft.ifftshift(np.angle(fshift))
img_back = np.fft.ifft2(f2shift)
#出来的是复数，无法显示
img_back = np.abs(img_back)
#调整大小范围便于显示
img_back = (img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))
plt.subplot(223),plt.imshow(img_back,'gray'),plt.title('仅仅包含相位',fontproperties=font_set)
plt.xticks([]),plt.yticks([])
plt.show()
#---------------------------------------------
# 逆变换--将两者合成看看
s1 = np.abs(fshift) #取振幅
s1_angle = np.angle(fshift) #取相位
s1_real = s1*np.cos(s1_angle) #取实部
s1_imag = s1*np.sin(s1_angle) #取虚部
s2 = np.zeros(img.shape,dtype=complex) 
s2.real = np.array(s1_real) #重新赋值给s2
s2.imag = np.array(s1_imag)

f2shift = np.fft.ifftshift(s2) #对新的进行逆变换
img_back = np.fft.ifft2(f2shift)
#出来的是复数，无法显示
img_back_compare = np.fft.ifft2(f2shift)
img_back = np.abs(img_back) # 断点运行,就是去实部呀
#调整大小范围便于显示
img_back = (img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))
plt.subplot(224),plt.imshow(img_back,'gray'),plt.title('another way',fontproperties=font_set)
plt.xticks([]),plt.yticks([])
plt.show()

# 从上面设置断点来看,其实每一个像素点[对图片而言]都对应一个a+b*i的虚函数,或者说cos(x)+i*sin(x),也就是说每一个像素点都有振幅A和相位角决定这个像素点的属性










