# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:47:53 2017

@author: rd0348
"""

import numpy as np
from matplotlib.pyplot import plot, show
# 快速傅里叶变换
x = np.linspace(0, 2 * np.pi, 30) #创建一个包含30个点的余弦波信号
wave = np.cos(x)
transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))  #对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
plot(transformed)  #使用Matplotlib绘制变换后的信号。
show()

# 移频
x = np.linspace(0, 2 * np.pi, 30) 
wave = np.cos(x)  #创建一个包含30个点的余弦波信号。
transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
shifted = np.fft.fftshift(transformed) #使用fftshift函数进行移频操作。
print(np.all((np.fft.ifftshift(shifted) - transformed) < 10 ** -9))  #用ifftshift函数进行逆操作，这将还原移频操作前的信号。
plot(transformed, lw=2)
plot(shifted, lw=3)
show()    #使用Matplotlib分别绘制变换和移频处理后的信号。

# single frequency signal
sampling_rate = 2**14
fft_size = 2**12
t = np.arange(0, 1, 1.0/sampling_rate)
x = np.array(map(lambda x : x*1e3, t))
y = np.sqrt(2)*np.sin(2*np.pi*1000*t)
y = y + 0.005*np.random.normal(0.0,1.0,len(y))
# fft
ys = y[:fft_size]
yf = np.fft.rfft(ys)/fft_size
freq = np.linspace(0,sampling_rate/2, fft_size/2+1)
freqs = np.array(map(lambda x : x/1e3, freq))
yfp = 20*np.log10(np.clip(np.abs(yf),1e-20,1e100))

print("*"*60)
x = np.random.rand(8)
print(x)
xf = np.fft.fft(x) #傅里叶变换
#注意ifft的运算结果实际上是和x相同的,可能有浮点细微差异
np.fft.ifft(xf) #逆傅里叶变换









