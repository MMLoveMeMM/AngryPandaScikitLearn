# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:38:09 2017

@author: rd0348
"""

import numpy as np
import matplotlib.pyplot as plt
m,n=(5,3)
x=np.linspace(0,1,m) #生成向量
y=np.linspace(0,1,n)
X,Y=np.meshgrid(x,y) #向量转换成矩阵
print(X.shape)
print(Y.shape)
plt.plot(X,Y,marker='.',color='blue',linestyle='none')
plt.show()
#获取坐标点坐标值
z=[i for i in zip(X.flat,Y.flat)]
print(z)