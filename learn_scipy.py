# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:11:49 2017

@author: rd0348
http://blog.csdn.net/pipisorry/article/details/51106570
"""

import numpy as np
from scipy.optimize import leastsq
import pylab as pl
import scipy.optimize as opt
from math import sin,cos
from scipy.optimize import fsolve
from scipy import integrate
from scipy import Parameters
from scipy.optimize import minimize
# 假设有一组实验数据(xi，yi ), 事先知道它们之间应该满足某函数关系yi=f(xi)，
# 通过这些已知信息，需要确定函数f的一些参数。
# 例如，如果函数f是线性函数f(x)=kx+b,那么参数 k和b就是需要确定的值。
# 如果用p表示函数中需要确定的参数，那么目标就是找到一组p,使得下面的函数S的值最小：
# 这种算法被称为最小二乘拟合(Least-square fitting)
def func(x,p):
    A,k,theta=p
    return A*np.sin(2*np.pi*k*x+theta)

def residuals(p,y,x):
    return y-func(x,p)

x=np.linspace(-2*np.pi,0,100)
print("x : \n",x)
A,k,theta=10,0.34,np.pi/6 # 真实数据的函数参数
y0 = func(x,[A,k,theta]) # 真实数据
y1=y0+2*np.random.randn(len(x)) # 加入噪声之后的实验数据
p0=[7,0.2,0] # 第一次猜测的函数拟合参数
# 调用leastsq进行数据拟合, residuals为计算误差的函数
# p0为拟合参数的初始值,# args为需要拟合的实验数据
plsq=leastsq(residuals,p0,args=(y1,x)) # residuals(p0,y1,x)
# 除了初始值之外，还调用了args参数，用于指定residuals中使用到的其他参数（直线拟合时直接使用了X,Y的全局变量）,
# 同样也返回一个元组，第一个元素为拟合后的参数数组；
# 这里将 (y1, x)传递给args参数。Leastsq()会将这两个额外的参数传递给residuals()。
# 因此residuals()有三个参数，p是正弦函数的参数，y和x是表示实验数据的数组。
print("真是参数 : ",[A,k,theta])
print("拟合参数 : ",plsq[0])

pl.plot(x, y0, label=u"真实数据")
pl.plot(x, y1, label=u"带噪声的实验数据")
pl.plot(x, func(x, plsq[0]), label=u"拟合数据")
pl.legend()
pl.show()

# ----------------------------------------------------------------
# optimize库提供了几个求函数最小值的算法
def test_fmin_convolve(fminfunc,x,h,y,yn,x0):
    
    def convolve_func(h):
        return np.sum((yn-np.convolve(x,h))**2)
    
    h0=fminfunc(convolve_func,x0)
    print(fminfunc.__name__)
    print("<--------------------------->")
    
    print("输出x*h0和y之间的相对误差 : \n",np.sum((np.convolve(x,h0)-y)**2)/np.sum(y**2))
    print("输出h0和h之间的相对误差 : \n",np.sum((h0-h)**2)/np.sum(h**2))
    
def test_n(m,n,nscale):
    x=np.random.rand(m)
    h=np.random.rand(n)
    y=np.convolve(x,h)
    yn=y+np.random.rand(len(y))*nscale
    x0=np.random.rand(n)
    
    test_fmin_convolve(opt.fmin, x, h, y, yn, x0)  
    test_fmin_convolve(opt.fmin_powell, x, h, y, yn, x0)  
    test_fmin_convolve(opt.fmin_cg, x, h, y, yn, x0)  
    test_fmin_convolve(opt.fmin_bfgs, x, h, y, yn, x0)
    
test_n(200,20,0.1)

# 非线性方程组求解
# func(x)是计算方程组误差的函数，它的参数x是一个矢量，表示方程组的各个未知数的一组可能解，
# func返回将x代入方程组之后得到的误差；x0为未知数矢量的初始值
def f(x):
    x0=float(x[0])
    x1=float(x[1])
    x2=float(x[2])
    return [
            5*x1+3,
            4*x0*x0-2*sin(x1*x2),
            x1*x2-1.5]
result = fsolve(f,[1,1,1])
print("fsolve result : ",result)
print("f(result) : ",f(result))

# 数值积分是对定积分的数值求解，
# 例如可以利用数值积分计算某个形状的面积
def half_circle(x):
    return (1-x**2)**0.5

N=10000
x=np.linspace(-1,1,N)
dx=2.0/N
y=half_circle(x)
print("half circle : ",dx*np.sum(y)*2)

print("numpy half circle : ",np.trapz(y,x)*2)

pi_half,err=integrate.quad(half_circle,-1,1)
print("pi_half : ",pi_half*2)

# 非线性拟合求解
def residual(params,x,data):
    amp=params['amp']
    pshift=params['phase']
    freq=params['frequency']
    decay=params['decay']
    model=amp*sin(x*freq+pshift)*exp(-x*x*decay)
    return (data-model)

params=Parameters()
params.add('amp',value=10,vary=False)
params.add('decay',value=0.007,min=0.0)
params.add('phase',value=0.2)
params.add('frequency',value=3.0,max=10)

x=np.linspace(0,15,301)
data=(5.*np.sin(2*x-0.1)*np.exp(-x*x*0.025))+np.random.normal(size=len(x),scale=0.2)

out=minimize(residual,params,args=(x,data),method='leastsq')
print(out.params)




"""
scipy中的optimize子包中提供了常用的最优化算法函数实现。我们可以直接调用这些函数完成我们的优化问题。optimize中函数最典型的特点就是能够从函数名称上看出是使用了什么算法。

下面optimize包中函数的概览：

1.非线性最优化

fmin – 简单Nelder-Mead算法
fmin_powell – 改进型Powell法
fmin_bfgs – 拟Newton法
fmin_cg – 非线性共轭梯度法
fmin_ncg – 线性搜索Newton共轭梯度法
leastsq – 最小二乘
2.有约束的多元函数问题

fmin_l_bfgs_b —使用L-BFGS-B算法
fmin_tnc —梯度信息
fmin_cobyla —线性逼近
fmin_slsqp —序列最小二乘法
nnls —解|| Ax - b ||_2 for x>=0
3.全局优化

anneal —模拟退火算法
brute –强力法
4.标量函数

fminbound
brent
golden
bracket
5.拟合

curve_fit– 使用非线性最小二乘法拟合
6.标量函数求根

brentq —classic Brent (1973)
brenth —A variation on the classic Brent（1980）
ridder —Ridder是提出这个算法的人名
bisect —二分法
newton —牛顿法
fixed_point
7.多维函数求根

fsolve —通用
broyden1 —Broyden’s first Jacobian approximation.
broyden2 —Broyden’s second Jacobian approximation
newton_krylov —Krylov approximation for inverse Jacobian
anderson —extended Anderson mixing
excitingmixing —tuned diagonal Jacobian approximation
linearmixing —scalar Jacobian approximation
diagbroyden —diagonal Broyden Jacobian approximation
8.实用函数

line_search —找到满足强Wolfe的alpha值
check_grad —通过和前向有限差分逼近比较检查梯度函数的正确性
"""

















