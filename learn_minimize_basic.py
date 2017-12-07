# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:20:20 2017

@author: rd0348
参考 : http://www.cnblogs.com/NaughtyBaby/p/5590081.html
"""

import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
# -----------------------------------------------------------
f=lambda x:2*np.sin(x+3)-x
x=np.linspace(-5,5,1000)
y=f(x)
plt.plot(x,y) # 画f的线
plt.axhline(0,color='k') # 划一根横线

res = opt.bisect(f,-5,5) # -5到5区间能使y=0的x值,即为函数的根
print("函数的根 : ",res)
# ------------------------------------------------------------
plt.plot(x,y)
plt.axhline(0,color='k')
plt.scatter(res,[0],c='r',s=100); # 利用画散点函数画出根坐标点,标记为红色高亮显示
plt.show()

# -------------------------------------------------------------
# 函数求最小化[局部]
# 求最小值就是一个最优化问题。求最大值时只需对函数做一个转换，比如加一个负号，或者取倒数，就可转成求最小值问题。所以两者是同一问题。
f=lambda x:1-np.sin(x)/x # 隐藏函数
x=np.linspace(-20.,20.,1000)
y=f(x)
# 当初始值为 3 值，使用 minimize 函数找到最小值。minimize 函数是在新版的 scipy 里，取代了以前的很多最优化函数，是个通用的接口，背后是很多方法在支撑
x0=10 # 可以修改成其他数,会发现,这个最小值是局部的,并且是最近的那个最低点,并不是全局函数的最低点,比如修改成x0=10
xmin=opt.minimize(f,x0).x
print("")

plt.plot(x,y)
plt.scatter(x0,f(x0),marker='o',s=300)
plt.scatter(xmin,f(xmin),marker='v',s=300)
plt.xlim(-20,20)
plt.show()

# ----------------------------------------------------------------
# 求全局最优点,即全局最低点
x0 =16
from scipy.optimize import basinhopping # 使用basinhopping函数完成
xmin=basinhopping(f,x0,stepsize=5).x
plt.plot(x,y)
plt.scatter(x0,f(x0),marker='o',s=300)
plt.scatter(xmin,f(xmin),marker='v',s=300)
plt.xlim(-20,20)
plt.show()

# ---------------------------------------------------------------
# 求多元函数最小值
def g(X):
    x,y=X
    return (x-1)**4+5*(y-1)**2-2*x*y

X_opt=opt.minimize(g,(8,3)).x # minimize函数会有专门的介绍
print(X_opt)

fig,ax=plt.subplots(figsize=(6,4))
x_=y_=np.linspace(-1,4,100)
X,Y=np.meshgrid(x_,y_)
c=ax.contour(X,Y,g((X,Y)),50)
ax.plot(X_opt[0],X_opt[1],'r*',markersize=15)
ax.set_xlabel(r"$x_1$",fontsize=18)
ax.set_ylabel(r"$x_2$",fontsize=18)
plt.colorbar(c,ax=ax)
fig.tight_layout()

# 绘制3D,上面红星点对应3D曲面最低点位置
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
x_ = y_ = np.linspace(-1, 4, 100) 
X, Y = np.meshgrid(x_, y_)
surf = ax.plot_surface(X, Y, g((X,Y)), rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
cset = ax.contour(X, Y, g((X,Y)), zdir='z',offset=-5, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5);

