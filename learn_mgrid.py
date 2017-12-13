# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:58:52 2017

@author: rd0348
python 3D绘图
要绘制3D图,那么一般即是二元一次方程,或者J(K,B),即J是关于两个变化因素的函数(可连续,也可离散)
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数  
# 支持中文显示
from matplotlib.font_manager import FontProperties  
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 

# 显示3D坐标
print("绘制坐标 : ")
fig1=plt.figure()#创建一个绘图对象  
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  
fig2=plt.figure()#创建一个绘图对象  
ax=Axes3D(fig2)#用这个绘图对象创建一个Axes对象(有3D坐标)  
plt.show()#显示模块中的所有绘图对象 

print("-"*60)

# mgrid
'''''fig1=plt.figure()#创建一个绘图对象 
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)'''  
X=np.arange(-2,2,1)  
Y=np.arange(-2,2,1)#创建了从-2到2，步长为1的arange对象  
#至此X,Y分别表示了取样点的横纵坐标的可能取值  
print ("X为",X)  
print ("Y为",Y)  
#用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点  
X,Y=np.meshgrid(X,Y)  
print ("扩充后X为")  
print (X)  
print ("扩充后Y为")  
print (Y)  

# 绘制散点 <1>
print("-"*60)
print("绘制散点<1> : ")
k,b=np.mgrid[1:3:3j,4:6:3j]
f_kb=3*k**2+2*b+1 # 线性方程

k.shape=-1,1
b.shape=-1,1
f_kb.shape=-1,1 #统统转成9行1列

fig=p.figure()
ax=p3.Axes3D(fig)
ax.scatter(k,b,f_kb,c='r')
ax.set_xlabel('k')
ax.set_ylabel('b')
ax.set_zlabel('ErrorArray')
p.show()

print("-"*60)
# 绘制散点 <2>
print("绘制散点<2> : ")
x = np.random.normal(0, 2, 125)
y = np.random.normal(0, 2, 125)
z = np.random.normal(0, 2, 125)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z,c='g')
plt.show()

# 绘制线型
print("绘制线型 : ")
# 生成数据
x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
y = np.sin(x)
z = np.cos(x) # 创建 3D 图形对象 , 复数线条
fig = plt.figure()
ax = Axes3D(fig) # 绘制线型图
ax.plot(x, y, z) # 显示图
plt.show()

# 绘制柱状
print("绘制柱状 : ")
fig = plt.figure()
ax = Axes3D(fig) # 生成数据并绘图
x = [0, 1, 2, 3, 4, 5, 6]
for i in x:  
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
    z = abs(np.random.normal(1, 10, 10))  
    ax.bar(y, z, i, zdir='y', color=['r', 'g', 'b', 'y'])
plt.show()

# 绘制曲面
print("绘制曲面 : ")
def fun(x,y):  
    return np.power(x,2)+np.power(y,2)  
  
fig1=plt.figure()#创建一个绘图对象  
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  
X=np.arange(-2,2,0.1)  
Y=np.arange(-2,2,0.1)#创建了从-2到2，步长为0.1的arange对象  
#至此X,Y分别表示了取样点的横纵坐标的可能取值  
#用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点  
X,Y=np.meshgrid(X,Y)  
Z=fun(X,Y)#用取样点横纵坐标去求取样点Z坐标
plt.title("3D图总标题",fontproperties=font_set)#总标题  
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面,
# rstride和cstride表示行列隔多少个取样点建一个小面，cmap表示绘制曲面的颜色，在pylot.cm下有很多选项可以选择
ax.set_xlabel('x轴标题', color='r',fontproperties=font_set)  # 可以为x坐标注释，y/z同理
ax.set_ylabel('y轴标题', color='g',fontproperties=font_set)  
ax.set_zlabel('z轴标题', color='b',fontproperties=font_set)  # 给三个坐标轴注明  
plt.show()#显示模块中的所有绘图对象  

print("-"*60)

# 综合应用
# 创建 3D 图形对象
print("绘制混合图像 : ")
fig = plt.figure()
ax = Axes3D(fig) # 生成数据并绘制图 红色的正弦曲线
x1 = np.linspace(-3 * np.pi, 3 * np.pi, 500)
y1 = np.sin(x1)
ax.plot(x1, y1, zs=0, c='red') # 生成数据并绘制图 散点
x2 = np.random.normal(0, 1, 100)
y2 = np.random.normal(0, 1, 100)
z2 = np.random.normal(0, 1, 100)
ax.scatter(x2, y2, z2,c='g') # 绿色的散点
plt.show()

# 创建 1 张画布
fig = plt.figure()
# 向画布添加子图 1 
ax1 = fig.add_subplot(1, 2, 1, projection='3d') # 生成子图 1 数据
x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
y = np.sin(x)
z = np.cos(x) # 绘制第 1 张图
ax1.plot(x, y, z) #=============== 
# 向画布添加子图 2
ax2 = fig.add_subplot(1, 2, 2, projection='3d') # 生成子图 2 数据
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X ** 2 + Y ** 2) # 绘制第 2 张图
ax2.plot_surface(X, Y, Z, cmap=plt.cm.winter)
plt.show()
