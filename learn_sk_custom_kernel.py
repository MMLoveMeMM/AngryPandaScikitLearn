# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:43:28 2017

@author: rd0348
理论参考 : http://blog.csdn.net/nieson2012/article/details/51337656
实际中，我们会经常遇到线性不可分的样例，此时，我们的常用做法是把样例特征映射到高维空间中去(如上文2.2节最开始的那幅图所示，
映射到高维空间后，相关特征便被分开了，也就达到了分类的目的)；
但进一步，如果凡是遇到线性不可分的样例，一律映射到高维空间，那么这个维度大小是会高到可怕的(如上文中19维乃至无穷维的例子)。那咋办呢？
此时，核函数就隆重登场了，核函数的价值在于它虽然也是讲特征进行从低维到高维的转换，但核函数绝就绝在它事先在低维上进行计算，
而将实质上的分类效果表现在了高维上，也就如上文所说的避免了直接在高维空间中的复杂计算
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

iris=datasets.load_iris()
X=iris.data[:,:2]
Y=iris.target

def def_kernel(X,Y):
#                 (2  0)
#    k(X, Y) = X  (    ) Y.T
#                 (0  1)
    M=np.array([[2,0],[0,1.0]])
    return np.dot(np.dot(X,M),Y.T)

h=0.01

clf=svm.SVC(kernel=def_kernel)
clf.fit(X,Y)

x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Paired,edgecolors='k')
plt.title('')
plt.axis('tight')
plt.show()

