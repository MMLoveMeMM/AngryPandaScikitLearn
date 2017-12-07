# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:41:48 2017

@author: rd0348
梯度下降法和课程中用到的最小二乘法都可用于最小化误差平方和，但两者的实现思路是不一样的。
最小二乘是一种纯数学方法，把所有数据组成矩阵方程来解出未知参数，而梯度下降法是把误差平方和作为一个函数，
求极值，误差平方和包含了所有的数据.
梯度下降法用来最小化误差平方和，和求函数极值
"""

# 一维梯度下降
import sympy
import numpy as np
from sympy.abc import x,y

fsym=x**2+3*x-10

def gd(fsym,x0,step=0.01,tol=1e-5):
    x1=x0
    while True:
        x0=x1
        x1=x0-step*fsym.diff().subs(x,x0)
        if np.abs(x1-x0)<tol:
            break
    return x1

print("gradient : ",gd(fsym,-1))

# 多维梯度下降
def gd_2d(f,X,step=0.01,tol=1e-3):
    X0 = X
    def gradient(f,X):
        dx = float(f.diff(x).subs({x:X[0], y:X[1]}))
        dy = float(f.diff(y).subs({x:X[0], y:X[1]}))
        norm = (1./np.sqrt(dx**2+dy**2))
        return norm*np.array([dx,dy])
    
    while True:
        z0 = float(f.subs({x:X0[0], y:X0[1]}))
        X1 = X0 - step*gradient(f,X0)
        z1 = float(f.subs({x:X1[0], y:X1[1]}))
        if np.abs(z1 - z0) < tol:
            break
        X0 = X1
    
    return X1
def gsym(x,y):
    return (x-1)**4 + 5 * (y-1)**2 - 2*x*y

print("多维梯度 : ",gd_2d(gsym(x,y),[0,0]))

# 将上面使用minimize函数实现
def g(X):
    x,y=X
    return (x-1)**4 + 5 * (y-1)**2 - 2*x*y

X_opt=opt.minimize(g,(8,3)).x
print("minimize : ",X_opt)


