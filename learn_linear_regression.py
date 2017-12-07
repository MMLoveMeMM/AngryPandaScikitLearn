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
    
    
if '__main__' == __name__:
    # 加载数据
    X, Y = load_data('abalone.txt')
    X, Y = standarize(X), standarize(Y)
    w = std_linearreg(X, Y)
    Y_prime = X*w

    print('w: {}'.format(w))

    # 计算相关系数
    corrcoef = get_corrcoef(np.array(Y.reshape(1, -1)),
                            np.array(Y_prime.reshape(1, -1)))
    print('Correlation coeffient: {}'.format(corrcoef))

    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    ## 绘制数据点
    #x = X[:, 1].reshape(1, -1).tolist()[0]
    #y = Y.reshape(1, -1).tolist()[0]
    #ax.scatter(x, y)

    ## 绘制拟合直线
    #x1, x2 = min(x), max(x)
    #y1 = (np.matrix([1, x1])*w).tolist()[0][0]
    #y2 = (np.matrix([1, x2])*w).tolist()[0][0]
    #ax.plot([x1, x2], [y1, y2], c='r')

    #plt.show()









