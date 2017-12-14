# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:35:13 2017

@author: rd0348
"""
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import urllib

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
# 观察一下数据集，X 有四个属性，y 有 0，1，2 三类
print("样本数据 : \n",iris_X,"\n目标值 : \n",iris_Y)
print(iris_X.shape)
print("样本数量 : ",iris_X.shape[0],"维度 : ",iris_X.shape[1])
X=iris_X[:,2]
print("获取数据集第二列集合 : \n",X)
X02=iris_X[:,0:2]
print("获取数据集第一列和第二列集合 : \n",X02)
# print("data第二列数据 : ",X)
# 把数据集分为训练集和测试集，其中 test_size=0.3，
# 即测试集占总数据的 30% ,也就是说训练集占70%
# 分开后的数据集，顺序也被打乱，这样更有利于学习模型
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_Y, test_size=0.3)

print("X_train : \n",X_train)
print("X_test : \n",X_test)
print("y_train : \n",y_train)
print("y_test : \n",y_test)

print("*"*60)
print("利用KNN模型开始预测 : ")
knn = KNeighborsClassifier() # 建立KNN模型
knn.fit(X_train, y_train) # 训练后就可以开始预测
pred_y=knn.predict(X_test)
print("训练后预测的值 : \n",pred_y)
print("实际对应的值 : \n",y_test)

# url with dataset
print("*"*60)
print("网络数据加载 : ")
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
# 这个写法注意:python3.5的写法,如果是2.7 : urllib.urlopen(url) 或者urllib2.urlopen(url) 这个要导入urllib2
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X17 = dataset[:,1:7]
y = dataset[:,8]
print("打印第一到第七列X : \n",X17)

print("*"*60)
print("本地文件数据加载 : ")
# read the first 4 columns
# 如果没有,可以到http://aima.cs.berkeley.edu/data/iris.csv下载这个数据文件
localfn="iris.csv"
data = genfromtxt(localfn,delimiter=',',usecols=(0,1,2,3))
print("iris数据的属性 : ",data.shape)
# read the fifth column
target = genfromtxt(localfn,delimiter=',',usecols=(4),dtype=str)
print("iris目标数据的属性 : ",target.shape)
print("获取iris文件的前四列数据 : \n",data)
print("获取iris文件的第四列目标数据 : \n",target)

print("*"*60)
print("本地数据库加载 : ")
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print("iris目标数据的属性 : ",iris.data.shape)
print("获取iris文件的前四列数据 : \n",iris_X)
print("获取iris文件的第四列目标数据 : \n",iris_Y)




