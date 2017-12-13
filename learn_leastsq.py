# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:52:33 2017

@author: rd0348
 最小二乘算法
 参考 : https://www.cnblogs.com/NanShan2016/p/5493429.html
 该算法也是一种用于求解最优系数的方法,通minimize()类似
"""
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 下面是一次线性方程利用最小二乘法求解
# 样本点
Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])

# 线性方程 : y=k*x+b
def func(p,x):
    k,b=p
    return k*x+b;

def offset(p,x,y,s):
    print(s)
    return func(p,x)-y
    
p0=[100,2]

s="注意这一句打印了多少次,就是尝试了多少次才能求解最优系数的 : "
para=leastsq(offset,p0,args=(Xi,Yi,s)) # 求解
k,b=para[0] # 求出了线性方程的最佳系数
print("k=",k,"b=",b); # 这里大概反复尝试9次,即可得到最优系数k,b
# 绘图形象说明
plt.scatter(Xi,Yi) # 绘制散点
p=np.array([k,b])
plt.plot(Xi,func(p,Xi),'b',lw=2)
plt.xlim(2,10)
plt.xlabel(r"$x$",fontsize=18)
plt.ylabel(r"$f(x,\beta)$",fontsize=18)
plt.show()

print("-"*60)
# 下面是二次线性方程利用最小二乘法求解
# 样本点
X1i=np.array([0,1,2,3,-1,-2,-3])
Y1i=np.array([-1.21,1.9,3.2,10.3,2.2,3.71,8.7])
def func1(p,x):
    a,b,c=p
    return a*x**2+b*x+c;

def offset1(p,x,y,s):
    print(s)
    return func1(p,x)-y

p1=[5,2,10]

s1="注意这一句打印了多少次,就是尝试了多少次才能求解最优系数的 : "
para1=leastsq(offset1,p1,args=(X1i,Y1i,s1))
a,b,c=para1[0] # 求出了线性方程的最佳系数
print("a=",a,"\tb=",b,"\tc=",c)

# 绘图看情况
plt.figure(figsize=(8,6))
plt.scatter(X1i,Y1i,color="red",label="Sample Point",linewidth=3) #画样本点
x=np.linspace(-5,5,1000)
y=a*x**2+b*x+c
plt.plot(x,y,color="blue",label="Fitting Curve",linewidth=2) #画拟合曲线
plt.legend()
plt.show()

print("*"*60)

# 最小二乘法基本由来
# 采样点
Xi=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Yi=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])

"""part 1"""
###需要拟合的函数func及误差error###
def func(p,x):
    k,b=p
    return k*x+b

def offset2(p,x,y):
    return func(p,x)-y #x、y都是列表，故返回值也是个列表

p0=[1,2]

###最小二乘法求k0、b0###
Para=leastsq(offset2,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
k0,b0=Para[0]
print("k0=",k0,"\tb0=",b0)

"""part 2"""
###定义一个函数，用于计算在k、b已知时，∑((yi-(k*xi+b))**2)###
def S(k,b):
    ErrorArray=np.zeros(k.shape) # k的shape事实上同时也是b的shape
    for x,y in zip(Xi,Yi): # zip(Xi,Yi)=[(8.19,7.01),(2.72,2.78),...,(3.78,4.05)]
        ErrorArray+=(y-(k*x+b))**2 
    return ErrorArray

###绘制ErrorArray+最低点###
# mayavi库没有安装成功,换下面的库显示
import mpl_toolkits.mplot3d
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
#画整个Error曲面
# 运行结果中蓝色部分曲面即是k,b
k,b=np.mgrid[k0-1:k0+1:10j,b0-1:b0+1:10j]
Err=S(k,b)

ax=plt.subplot(111,projection='3d')
ax.plot_surface(k,b,Err/500.0,rstride=1,cstride=1, alpha=0.5)
Err0=S(k0,b0) # 上面已经获取了最优点,那么就把最优系数代入方程,就可以得到最优点系数(K,B)坐标位置
ax.scatter(k0,b0,Err0/500.0,c='r') # 画出最优点,注意看曲面上那个红点,刚好在曲面最低位置
ax.set_xlabel('k')
ax.set_ylabel('b')
ax.set_zlabel('ErrorArray')
p.show()




