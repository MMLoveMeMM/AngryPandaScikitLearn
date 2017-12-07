# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:01:34 2017

@author: rd0348
该单元 : 多项式最优求解
"""
import sympy
from sympy.abc import x
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
# 二分法求解
# 对于区间[a，b]上连续不断且f（a）·f（b）<0的函数y=f（x），
# 通过不断地把函数f（x）的零点所在的区间一分为二，使区间的两个端点逐步逼近零点，进而得到零点近似值的方法叫二分法
# 二分法具体定义参考 : https://baike.baidu.com/item/%E4%BA%8C%E5%88%86%E6%B3%95/1364267?fr=aladdin
f=lambda x:x**2+3*x-10
def halfinterval(f,a,b,eps):
    midValue=(a+b)/2.0
    
    if f(a)==0:
        root=a
    elif f(b)==0:
        root=b
    elif f(a)*f(b)>0:
        root=midValue
    elif f(a)*f(midValue)<0:
        if(midValue-a)<eps:
            root=(midValue+a)/2.0
        else:
            root=halfinterval(f,a,midValue,eps)
    else:
        root = halfinterval(f,midValue,b,eps) 
    return root
            
res = halfinterval(f,10,-6,0.01)
print("二分法求解 : ",res)

# 用SymPy 库实现多项式的求解
f=x**2+3*x-10
ret = sympy.solve(f)
print("Sympy 求解多项式 : ",ret)

# 使用scipy.optimize 中的bisect或者brentq求根
# print(opt.bisect(f,-10,0),opt.bisect(f,0,5))
# print(opt.brentq(f,-10,0),opt.brentq(f,0,5))

f=lambda x: x**2+3*x-10
x=np.linspace(-10,10,1000)
y=f(x)
plt.plot(x,y)
plt.axhline(0,color='k')

