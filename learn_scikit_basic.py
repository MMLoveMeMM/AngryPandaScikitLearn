# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:49:33 2017

@author: rd0348
"""

import numpy as np
import urllib
# url with dataset : 加载数据
print("*"*60)
print("url with dataset : 加载数据")
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data =urllib.request.urlopen(url)#.read() 
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]

# Data Normalization : 数据归一化
print("*"*60)
print("Data Normalization : 数据归一化")
from sklearn import preprocessing
# normalize the data attributes
print("归一化之前X : ",normalized_X)
normalized_X = preprocessing.normalize(X)
print("归一化之后normalized_X : ",X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

# Feature Selection : 特征选取
print("*"*60)
print("Feature Selection : 特征选取")
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

# Logistic Regression : 逻辑线性回归
print("*"*60)
print("Logistic Regression : 逻辑线性回归")
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Naive Bayes : 贝叶斯
print("*"*60)
print("Naive Bayes : 贝叶斯")
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# k-Nearest Neighbours : k-最近邻
print("*"*60)
print("k-Nearest Neighbours : k-最近邻")
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Decision Trees : 决策树
print("*"*60)
print("Decision Trees : 决策树")
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Support Vector Machines : 支持向量机
from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))






