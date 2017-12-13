# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:34:26 2017

@author: rd0348
参考 : http://www.cnblogs.com/NaughtyBaby/p/5590081.html
minimize()函数本质 : 对于一个给定的线性方程(形式)和给点散点,根据散点求解线性方程的系数项的值
求解原理 : 将所有的散点距离预测点的误差的平方和[组],选择平方和组中最小的那个平方和对应系数项返回.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
# 算出到[1,1],[4,6],[9,2]的距离最小的坐标
# 最小的距离即假设一点[x,y]到上面三个坐标的平方和最小值
# 其实这个也是后面线性回归的代价函数[损失函数],下同
def f(coord,x,y):
    return np.sum((coord[0]-x)**2+(coord[1]-y)**2) # 坐标相减的平方和

x=np.array([1,4,9])
y=np.array([1,6,2])

initial=np.array([50,5]) # 起始点,没有太大意义
print(f(initial,x,y))

res=minimize(f,initial,args=(x,y)) # 根据f函数,返回距离上面三个坐标最小的点的坐标
print("res.x : ",res.x)
print(f(res.x,x,y))

print("-"*60)
print("绘制离已知三个点最优的点[橙色点] : ")
plt.scatter(x,y) # 绘制散点
plt.plot(x,y,'r',lw=2) # 绘制实际线,红色线,宽度为2
plt.scatter(res.x[0],res.x[1]) # 绘制拟合点,平方和最小点
plt.show()

# 下面添加一个限制条件,即minimize函数最后一个参数是可以再添加额外的限制条件的
print("-"*60)
print("在限定条件下,绘制离已知三个点最优的点[橙色点] : ")
cons ={'type':'eq','fun':lambda coord:coord[0]**2+coord[1]**2-100}
# 其实它的算法就是,根据给出的散点,求出散点的拟合点
res=minimize(f,initial,args=(x,y),constraints=cons)
print("res.x : ",res.x)
print(f(res.x,x,y))
plt.scatter(x,y) # 绘制散点
plt.plot(x,y,'r',lw=2) # 绘制实际线,红色线,宽度为2
plt.scatter(res.x[0],res.x[1]) # 绘制拟合点,平方和最小点
plt.show()
# -----------------------------------------------------------
# 根据上面结合一个简单的实际看看minimize函数
N=50 # 产生50个样本散点
m_true=2
b_true=-1
dy=2.0
np.random.seed(0)
xdata=10*np.random.rand(N)
# 散点y坐标根据y=b_true+m_true*xdata这个线性做参考算出,但是y点有一定的随机性,
# 这个假设是先知道了m_true和b_true两个值
ydata=np.random.normal(b_true+m_true*xdata,dy)

plt.errorbar(xdata,ydata,dy,fmt='.k',ecolor='lightgray');

# --------------------------------------------------------------
# y = theta[0]+theta[1]*x
# y-(theta[0]+theta[1]*x)
# 下面是算出他们的误差平方差
# 误差平方和大，表示真实的点和预测的线之间距离太远，说明拟合得不好，偏离预测线越大
# 应该是使误差平方和最小，即最优的拟合线.

#损失函数[代价函数]
def chi2(theta,x,y):
    return np.sum(((y-theta[0]-theta[1]*x))**2) # 所有点误差的平方和

theta_guess=[0,1]
# 其实minimize函数说白了就是帮助求出最优theta值
theta_best=minimize(chi2,theta_guess,args=(xdata,ydata)).x # 算出平方和最小的对应的theta值
# 最优系数,是指拟合性最好的那根预计线
print("theta_best[最好的系数] : \n",theta_best)
# 然后上面求出的最优theta值带入chi2函数,获得最优[x,y]坐标
print("最好的误差平方和 : ",chi2(theta_best,xdata,ydata)) # 打印最好的拟合预测线对应最小的偏差平方之和

xfit=np.linspace(0,10)
yfit=theta_best[0]+theta_best[1]*xfit

plt.errorbar(xdata,ydata,dy,fmt='.k',ecolor='lightgray')
plt.plot(xfit,yfit,'-k')

# 另外一种采用最小二乘least square,这里只是做一个基本的引入
# 上面用的是 minimize 方法，这个问题的目标函数是误差平方和，这就又有一个特定的解法，即最小二乘。
# 最小二乘的思想就是要使得观测点和估计点的距离的平方和达到最小，这里的“二乘”指的是用平方来度量观测点与估计点的远近
# 在古汉语中“平方”称为“二乘”），“最小”指的是参数的估计值要保证各个观测点与估计点的距离的平方和达到最小。
# 关于最小二乘估计的计算，涉及更多的数学知识，这里不想详述，其一般的过程是用目标函数对各参数求偏导数，并令其等于 0，得到一个线性方程组
def deviations(theta,x,y):
    return (y-theta[0]-theta[1]*x)
theta_best,ier=leastsq(deviations,theta_guess,args=(xdata,ydata))
print("leastsq[最优系数] : ",theta_best)

# 非线性最小二乘
print("-"*60)
print("非线性最小二乘 : ")
def f(x,beta0,beta1,beta2):
    return beta0+beta1*np.exp(-beta2*x**2)

beta=(0.25,0.75,0.5) # 猜测的3个系数
xdata=np.linspace(0,5,50)
y=f(xdata,*beta)
ydata=y+0.05*np.random.randn(len(xdata))

def g(beta):
    return ydata-f(xdata,*beta)

beta_start=(1,1,1)
beta_opt,beta_cov=leastsq(g,beta_start)
print("非线性最优系数 : ",beta_opt)

fig,ax=plt.subplots()
ax.scatter(xdata,ydata) # 绘制散点
ax.plot(xdata,y,'r',lw=2) # 绘制实际线,红色线,宽度为2
ax.plot(xdata,f(xdata,*beta_opt),'b',lw=2) # 根据最优系数,绘制最优拟合线,蓝色,线宽为2
ax.set_xlim(0,5)
ax.set_xlabel(r"$x$",fontsize=18)
ax.set_ylabel(r"$f(x,\beta)$",fontsize=18)
fig.tight_layout()

# --------------------------------------------------------------------
# 有约束的最小化
# 有约束的最小化是指，要求函数最小化之外，还要满足约束条件，举例说明.
print("-"*60)
print("有约束的最小化 : ")
def f(X):
    x,y=X
    return (x-1)**2+(y-1)**2

x_opt = minimize(f,(0,0),method='BFGS')
# 假设有约束条件，x 和 y 要在一定的范围内，如 x 在 2 到 3 之间，y 在 0 和 2 之间
bnd_x1,bnd_x2=(0,3),(0,2) # 这个区域在显示在结果的暗色方形
x_cons_opt = minimize(f, np.array([-1, -1]), method='L-BFGS-B', bounds=[bnd_x1, bnd_x2]).x # bounds 矩形约束
print("x_cons_opt",x_cons_opt)
print("x_cons_opt[0]",x_cons_opt[0])
print("x_cons_opt[1]",x_cons_opt[1])
fig, axx = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = axx.contour(X, Y, f((X,Y)), 50)
axx.scatter(x_cons_opt[0], x_cons_opt[1]) # 没有约束下的最小值，红色五角星 
axx.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15) # 有约束下的最小值，红色星星
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]), 
                           bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
                           facecolor="grey")
axx.add_patch(bound_rect)
axx.set_xlabel(r"$x_1$", fontsize=18)
axx.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()

# 说白了就是求出最优系数,然后带入线性方程,输入要预测的根据值,输出最终的预测值
# 这样在后面线性回归里面,就可以根据上面的继续做预测操作了

