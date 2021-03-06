# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:54:04 2017

@author: rd0348
理论参考 : http://www.blogjava.net/zhenandaci/category/31868.html
http://blog.csdn.net/nieson2012/article/details/51337656
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 创建40个独立的点
X,y=make_blobs(n_samples=40,centers=2,random_state=6)
# 选择适配模型
clf=svm.SVC(kernel='linear',C=1000)
clf.fit(X,y)
# 绘制散点图
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=plt.cm.Paired)

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
# 计算间隔距
YY,XX=np.meshgrid(yy,xx) # 这个地方注意,得到的是点的集合,将点画出来,这些点就组成了直线
xy=np.vstack([XX.ravel(),YY.ravel()]).T 
Z=clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,
           linestyles=['--','-','--'])
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,
           linewidth=1,facecolors='none')
plt.show()
