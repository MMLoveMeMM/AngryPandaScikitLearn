# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:19:58 2017

@author: rd0348
"""
# 补充内容
# 画散点
import matplotlib.pyplot as plt
import numpy as np

n=50
x=np.random.rand(n)*2
y=np.random.rand(n)*2
colors=np.random.rand(n)
area=np.pi*(10*np.random.rand(n))**2
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker=(9,3,30))
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='.')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker=',')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='o')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='v')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='^')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='<')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='>')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='1')
# plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='2')
plt.scatter(x,y,s=area,c=colors,alpha=0.5,marker='3')
plt.show()

# numpy 设计向量矩阵运算的函数
#mean()函数功能：求取均值
#经常操作的参数为axis，以m * n矩阵举例：
#axis 不设置值，对 m*n 个数求均值，返回一个实数
#axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
#axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
arr1=np.array([[1,2,3],[2,3,4],[4,5,6],[7,8,9]])
mx=np.mat(arr1)
print("mx : \n",mx)
# 对所有的元素求均值
print("矩阵所有元素的均值: \n",np.mean(mx))
print("压缩行,对各列求均值 : \n",np.mean(mx,0))
print("压缩列,对各行求均值 : \n",np.mean(mx,1))

# std()函数计算标准差
stdnum=np.array([[1,2],[3,4]])
print("计算全局标准差 : \n",np.std(stdnum))
print("axis=0计算每一列的标准差 : \n",np.std(stdnum,axis=0))
print("axis=1计算每一行的标准差 : \n",np.std(stdnum,axis=1))

# var()方差
varnum=np.array([[1,2,3],[3,4,5]])
print("方差 : \n",np.var(varnum))


#diff函数返回一个由相邻数组元素的差值构成的数组
# [2-1,3-2,7-3,5-7]
diffnum=np.array([1,2,3,7,5])
print("diff : \n",np.diff(diffnum))

#exp函数
expnum=np.array([1,2,5,1,4])
print("exp : \n",np.exp(expnum))

#median() 函数
mednum=np.array([3,4,5,3,2])
print("返回中位数 : \n",mednum)

#linspace(s,e,[n])函数起始s,终止e,个数n(可选),返回一个元素值在指定范围内均匀分布的数组
# [-1,0]之间均匀取出5个值
print("linspace : \n",np.linspace(-1,0,5))












