# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:40:19 2017

@author: rd0348
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# -----------------------------------------------------------------------------
#随机排列交叉验证
#ShuffleSplit 可以定义划分迭代次数和训练测试集的划分比例
from sklearn.model_selection import ShuffleSplit
X=np.arange(5)
ss=ShuffleSplit(n_splits=3,test_size=0.25,random_state=0)#random_state保证了随机的可再现性
for train_index,test_index in ss.split(X):
    print("随机样本划分 : %s %s" %(train_index,test_index))
    
# KFold 
#对于假设其独立同分布的数据
#KFold 将样本划分为k组，如果k=n，就是所谓的留一法。
import numpy as np
from sklearn.model_selection import KFold
X=["a","b","c","d"]
kf=KFold(n_splits=2)
for train,test in kf.split(X):
    print("KFold : %s %s" %(train,test))#生成的折用下标表示原始数据位置
    
# LOO
X=np.array([[0.,0.],[1.,1.],[-1.,-1.],[2.,2.]])
y=np.array([0,1,0,1])
X_train,X_test,y_train,y_test=X[train],X[test],y[train],y[test]

# Leave One Out (LOO) 留一法
from sklearn.model_selection import LeaveOneOut
X=[1,2,3,4]
loo=LeaveOneOut()
for train,test in loo.split(X): # 将训练集X进行划分
    print("LOO : %s %s"%(train,test))

#Leave P Out (LPO) 与LeaveOneOut相似，从n个样本中选出p个样本的排列
from sklearn.model_selection import LeavePOut
X=np.ones(4)
lpo=LeavePOut(p=2)
for train,test in lpo.split(X): # 将训练集X进行划分
    print("LPO : %s %s" %(train,test))

# ------------------------------------------------------------------------------------------------
# 上面列举了部分数据划分的办法
# 那如何对划分的合理性进行评估呢?
# 加载数据库数据
iris=datasets.load_iris()
print("数据属性 : ",iris.data.shape)
print("目标数据属性 : ",iris.target.shape)

# 划分训练集和测试集
# test_size=0.4 : 将整个样本的60%作为训练样本,将40%作为测试样本
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
X_train.shape,y_train.shape
print("划分训练集和测试集后训练数据属性 : ",X_train.shape)
print("划分训练集和测试集后训练目标数据属性 : ",y_train.shape)

print("划分训练集和测试集后测试数据属性 : ",X_test.shape)
print("划分训练集和测试集后测试目标数据属性 : ",y_test.shape)
# 分好上面的样本得到训练样本
clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
# 测试准确率
ret = clf.score(X_test,y_test)
print("ret : ",ret)

from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear',C=1)
scores=cross_val_score(clf,iris.data,iris.target,cv=5)#5折交叉验证
print("scores : ",scores)
#平均值和95%的置信区间可以计算得出
print("Accuary: %0.2f(+/-%0.2f)" % (scores.mean(),scores.std()*2))

#默认情况下，每次CV迭代计算score是估计量的得分方法。可以通过使用评分参数来改变其评分参数：
from sklearn import metrics
scores=cross_val_score(clf,iris.data,iris.target,cv=5,scoring='f1_macro')
print("scores : ",scores)













