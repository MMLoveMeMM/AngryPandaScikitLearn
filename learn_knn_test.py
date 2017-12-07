# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:42:51 2017

@author: rd0348
"""
import sys
import learn_knn
from numpy import *
dataSet,labels = learn_knn.createDataSet()
inputdata = array([0.5,2.3])
K = 5
output = learn_knn.classify(inputdata,dataSet,labels,K)
print("测试数据为:",inputdata,"分类结果为：",output)