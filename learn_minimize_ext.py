# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:12:32 2017

@author: rd0348
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 不等式约束
def f(X):
    return (X[0] - 1)**2 + (X[1] - 1)**2

def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4

x_opt = minimize(f, (0, 0), method='BFGS').x
constraints = [dict(type='ineq', fun=g)] # 约束采用字典定义，约束方式为不等式约束，边界用 g 表示
x_cons_opt = minimize(f, (0, 0), method='SLSQP', constraints=constraints).x
fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f((X, Y)), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15) # 蓝色星星，没有约束下的最小值

ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color="grey")
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15) # 在区域约束下的最小值

ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()