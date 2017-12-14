# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:16:34 2017

@author: rd0348

SVM既可以用来分类，就是SVC；又可以用来预测，或者成为回归，就是SVR
概念二 : 
    交叉验证也称为CV。CV是用来验证分类器的性能一种统计分析方法，基本思想就是对原始数据(dataset)进行分组，
    一部分做为训练集(train set)，另一部分做为验证集(validation set)，首先用训练集对分类器进行训练，
    再利用验证集来测试训练得到的模型(model)，以此来做为评价分类器的性能指标
"""
from sklearn.svm import SVR  
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
import matplotlib.pyplot as plt


X=np.array([[-1,-1],[-2,-1],[1,1,],[2,1]])
y=np.array([1,1,2,2])
print("*"*60)
print("SVC 分类")
clf=SVC()
clf.fit(X,y)
print(clf.fit(X,y))
print("SVC预测结果精度 : ",clf.predict([[-0.8,-1]]))

print("*"*60)
print("NuSVC 分类")
clf=NuSVC()
clf.fit(X,y)
print(clf.fit(X,y))
print("NuSVC预测结果精度 : ",clf.predict([[-0.8,-1]]))

# 这里目前的阶段只看看SVR,支持向量机后面研究
print("SVR 线性回归") 
# Generate sample data  
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # 产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列  
y = np.sin(X).ravel()   # np.sin()输出的是列，和X对应，ravel表示转换成行  

# Add noise to targets  
y[::5] += 3 * (0.5 - np.random.rand(8))  

# Fit regression model  
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  
svr_lin = SVR(kernel='linear', C=1e3)  
svr_poly = SVR(kernel='poly', C=1e3, degree=2)  
y_rbf = svr_rbf.fit(X, y).predict(X)  
y_lin = svr_lin.fit(X, y).predict(X)  
y_poly = svr_poly.fit(X, y).predict(X)  

# look at the results  
lw = 2  
plt.scatter(X, y, color='darkorange', label='data')  
plt.hold('on')  
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')  
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')  
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')  
plt.xlabel('data')  
plt.ylabel('target')  
plt.title('Support Vector Regression')  
plt.legend()  
plt.show()




"""
SVC参数解释 
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
（6）probablity: 可能性估计是否使用(true or false)； 
（7）shrinking：是否进行启发式； 
（8）tol（default = 1e - 3）: svm结束标准的精度; 
（9）cache_size: 制定训练所需要的内存（以MB为单位）； 
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
（11）verbose: 跟多线程有关，不大明白啥意思具体； 
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None 
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 
 ps：7,8,9一般不考虑

NuSVC参数 
nu：训练误差的一个上界和支持向量的分数的下界。应在间隔（0，1 ].
其余同SVC

LinearSVC 参数解释  
C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；  
loss ：指定损失函数  
penalty ：  
dual ：选择算法来解决对偶或原始优化问题。当n_samples > n_features 时dual=false。  
tol ：（default = 1e - 3）: svm结束标准的精度;  
multi_class：如果y输出类别包含多类，用来确定多类策略， ovr表示一对多，“crammer_singer”优化所有类别的一个共同的目标  
如果选择“crammer_singer”，损失、惩罚和优化将会被被忽略。  
fit_intercept ：  
intercept_scaling ：  
class_weight ：对于每一个类别i设置惩罚系数C = class_weight[i]*C,如果不给出，权重自动调整为 n_samples / (n_classes * np.bincount(y))  
verbose：跟多线程有关，不大明白啥意思具体
"""















