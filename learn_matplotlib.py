# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:51:06 2017

@author: rd0348
"""

import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
plt.style.use('ggplot')
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

# ---基本绘图---
# 坐标轴范围设置及其显示

# 产生一个序列数组,间隔差为0.02
x=np.arange(-5.0,5.0,0.02)
# print("打印x序列 : \n",x)
y1=np.sin(x)

#绘图配置
plt.figure(1)
plt.subplot(211)
plt.plot(x,y1)

# 与上面对比,缩小一个区间显示
plt.subplot(212)
#设置x轴的范围
xlim(-2.5,2.5)
#设置y轴的范围
ylim(-1,1)
plt.plot(x,y1)
plt.show()
# 用一条指令画多条不同格式线
t=np.arange(0.,5.,0.2)
# 'r--' : 这种一种mask,r : 曲线轨迹代表红色,"--" : 第一根代表破折线,第二根代表水平线
# 'bs' : b : 代表blue蓝色,s : 正方形[线条标记]
# 'g^' : g : 代表green绿色,'^' : 一角朝上的三角形
# 这些都有一个"线条相关属性标记设置"表格
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

# ---绘制直方图---
# 增加坐标轴文字内容,增加整个图形title,增加曲线说明文字

#产生数据
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# 数据的直方图
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

# 设置x轴描述文本
plt.xlabel('Smarts')
# 设置y轴描述文本
plt.ylabel('Probability')
# 添加标题
plt.title('Histogram of IQ')
# 添加文字[曲线显示区域内容添加文字]
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# 坐标轴限制xmin,xmax,ymin,ymax
plt.axis([40, 160, 0, 0.03])
# 添加栅格
plt.grid(True)
plt.show()

# ---给图像添加文本注释---
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
# 第一个参数 : 注释说明内容
# 第二个参数 : 热点坐标
# 第三个参数 : 注释文本左下角坐标位置[注释文本显示的位置]
# 第四个参数 : 箭头参数和属性设置
plt.annotate('[annotate] max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.ylim(-2,2)
plt.show()

# ---设置轴刻度文字---
# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
plt.figure(figsize=(8,6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
# 表示把图标分割成1*1的网格
# 第一个参数是行数，第二个参数是列数，第三个参数表示图形的标号
plt.subplot(1,1,1) # 这个也可以简写plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# 绘制正弦曲线，使用红色的、连续的、宽度为 1 （像素）的线条
plt.plot(X, S, color="r", lw=4.0, linestyle="-")
# 还是可以设置轴范围
plt.axis([-4,4,-1.2,1.2])

# 设置轴记号
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])
# 在屏幕上显示
plt.show()

# ---移动坐标轴位置---
# 对比上下两个坐标轴位置,一个坐标原点在左下方,一个在正中心
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

plt.legend(loc='upper left')
plt.show()

# --- 这个涉及到sklearn包[数据处理] 这个在后面更常见---
# 加载数据
boston = datasets.load_boston()
yb = boston.target.reshape(-1, 1)
Xb = boston['data'][:,5].reshape(-1, 1)
# 开始绘制设置
plt.scatter(Xb,yb)
plt.ylabel('value of house /1000 ($)')
plt.xlabel('number of rooms')
# 创建线性回归线
regr = linear_model.LinearRegression()
# 根据上面的开始训练这个模型
regr.fit( Xb, yb)
# 显示出来
plt.scatter(Xb, yb,  color='black')
plt.plot(Xb, regr.predict(Xb), color='blue',
         linewidth=3)
plt.show()

# ---后面接入这个库就可以是sklearn库学习了---
# 有了sklearn库才勉强开始进入大数据入门了
lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# --- 当然还有很多其他的,但是初级基本上足够了,如果遇到特别的,可以另行再查不迟---
