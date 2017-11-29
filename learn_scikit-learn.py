# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:58:08 2017

@author: rd0348
"""

# http://scikit-learn.org/stable/auto_examples/index.html#general-examples
"""
对Python语言有所了解的科研人员可能都知道SciPy——一个开源的基于Python的科学计算工具包。基于SciPy，目前开发者们针对不同的应用领域已经发展出了为数众多的分支版本，它们被统一称为Scikits，即SciPy工具包的意思。而在这些分支版本中，最有名，也是专门面向机器学习的一个就是Scikit-learn。

Scikit-learn项目最早由数据科学家 David Cournapeau 在 2007 年发起，需要NumPy和SciPy等其他包的支持，是Python语言中专门针对机器学习应用而发展起来的一款开源框架。

和其他众多的开源项目一样，Scikit-learn目前主要由社区成员自发进行维护。可能是由于维护成本的限制，Scikit-learn相比其他项目要显得更为保守。这主要体现在两个方面：一是Scikit-learn从来不做除机器学习领域之外的其他扩展，二是Scikit-learn从来不采用未经广泛验证的算法。

本文将简单介绍Scikit-learn框架的六大功能，安装和运行Scikit-learn的大概步骤，同时为后续各更深入地学习Scikit-learn提供参考。原文来自infoworld网站的特约撰稿人Martin Heller，他曾在1986-2010年间做过长达20多年的数据库、通用软件和网页开发，具有丰富的开发经验。

Scikit-learn的六大功能

Scikit-learn的基本功能主要被分为六大部分：分类，回归，聚类，数据降维，模型选择和数据预处理。

分类是指识别给定对象的所属类别，属于监督学习的范畴，最常见的应用场景包括垃圾邮件检测和图像识别等。目前Scikit-learn已经实现的算法包括：支持向量机（SVM），最近邻，逻辑回归，随机森林，决策树以及多层感知器（MLP）神经网络等等。

需要指出的是，由于Scikit-learn本身不支持深度学习，也不支持GPU加速，因此这里对于MLP的实现并不适合于处理大规模问题。有相关需求的读者可以查看同样对Python有良好支持的Keras和Theano等框架。

回归是指预测与给定对象相关联的连续值属性，最常见的应用场景包括预测药物反应和预测股票价格等。目前Scikit-learn已经实现的算法包括：支持向量回归（SVR），脊回归，Lasso回归，弹性网络（Elastic Net），最小角回归（LARS ），贝叶斯回归，以及各种不同的鲁棒回归算法等。可以看到，这里实现的回归算法几乎涵盖了所有开发者的需求范围，而且更重要的是，Scikit-learn还针对每种算法都提供了简单明了的用例参考。

聚类是指自动识别具有相似属性的给定对象，并将其分组为集合，属于无监督学习的范畴，最常见的应用场景包括顾客细分和试验结果分组。目前Scikit-learn已经实现的算法包括：K-均值聚类，谱聚类，均值偏移，分层聚类，DBSCAN聚类等。

数据降维是指使用主成分分析（PCA）、非负矩阵分解（NMF）或特征选择等降维技术来减少要考虑的随机变量的个数，其主要应用场景包括可视化处理和效率提升。

模型选择是指对于给定参数和模型的比较、验证和选择，其主要目的是通过参数调整来提升精度。目前Scikit-learn实现的模块包括：格点搜索，交叉验证和各种针对预测误差评估的度量函数。

数据预处理是指数据的特征提取和归一化，是机器学习过程中的第一个也是最重要的一个环节。这里归一化是指将输入数据转换为具有零均值和单位权方差的新变量，但因为大多数时候都做不到精确等于零，因此会设置一个可接受的范围，一般都要求落在0-1之间。而特征提取是指将文本或图像数据转换为可用于机器学习的数字变量。

需要特别注意的是，这里的特征提取与上文在数据降维中提到的特征选择非常不同。特征选择是指通过去除不变、协变或其他统计上不重要的特征量来改进机器学习的一种方法。

总结来说，Scikit-learn实现了一整套用于数据降维，模型选择，特征提取和归一化的完整算法/模块，虽然缺少按步骤操作的参考教程，但Scikit-learn针对每个算法和模块都提供了丰富的参考样例和详细的说明文档。
"""

# ---下面是所有机器学习资料的第一个需要研究的线性回归样例---
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()