# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:38:56 2017

@author: rd0348
"""
#
import numpy as np
import matplotlib.pyplot as plt
# 标准线性相关
xcord=[208,152,113,227,137,238,178,104,191,130]
ycord=[21.6,15.5,10.4,31.0,13.0,32.4,19.0,10.4,19.0,11.8]
plt.scatter(xcord,ycord,s=30,c='r',alpha=0.5,marker='3')
#plt.show()

# y=a*x+b
# 这里a,b的值直接指定了
a=0.1612;b=-8.6394
# 产生训练数据集
x=np.arange(90.0,250.0,0.1)
# 训练模型
y=a*x+b
# 显示效果
plt.plot(x,y)
plt.show()

# 运行完后,查看效果,再看看程序,a,b的值是怎么来的呢?
# 下面演示基本的求法
def load_data(filename):
    X,Y=[],[]
    with open(filename,'r') as f:
        for line in f:
            splited_line=[float(i) for i in line.split()]
            x,y=splited_line[:-1],splited_line[-1]
            X.append(x)
            Y.append(y)
    X,Y=np.matrix(X),np.matrix(Y).T
    return X,Y

def standarize(X):
    std_deviation=np.std(X,0) # 计算全局标准差
    mean=np.mean(X,0)
    return (X-mean)/std_deviation

def std_linearreg(X,Y):
    xTx=X.T*X
    if np.linalg.det(xTx)==0:
        print("xTx is a singular matrix")
        return 
    return xTx.I*X.T*Y
def get_corrcoef(X,Y):
    # X,Y协方差
    # 参考learn_math.py说明
    cov=np.mean(X*Y)-np.mean(X)*np.mean(Y)
    return cov/(np.var(X)*np.var(Y))**0.5
    
    
xcord,ycord=load_data("abalone.txt")
xcord,ycord=standarize(xcord),standarize(ycord)
w=std_linearreg(xcord,ycord)
y_prime=xcord*w
print("w {} : \n",format(w))

# 计算相关系数
corrcoef=get_corrcoef(np.array(ycord.reshape(1,-1)),np.array(y_prime.reshape(1,-1)))
print('Correlation coeffient : \n',format(corrcoef))










