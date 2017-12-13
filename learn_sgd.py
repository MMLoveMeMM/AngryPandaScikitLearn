# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:41:32 2017
learn_sgd.py
@author: rd0348
梯度求解:
梯度下降法（gradient descent），又名最速下降法（steepest descent）
是求解无约束最优化问题最常用的方法，它是一种迭代方法，每一步主要的操作是求解目标函数的梯度向量，
将当前位置的负梯度方向作为搜索方向
梯度下降法特点：越接近目标值，步长越小，下降速度越慢
梯度下降法分为:
    <1> : 批量梯度下降法
    <2> : 随机梯度下降法
网上面还有梯度下降(learn_gradient_descent_basic.py),小批量梯度下降
两种对应的推到公式区别 :
    <1> : theta.j = theta.j-(alpha/m)*(损失函数i)*Xi
    <2> : theta.j = theta.j-alpha*(损失函数i)*Xi
    其中theta.j是指第j项系数;m代表样本数量;Xi为样本向量集
    
下面是介绍随机梯度下降法
优点：训练速度快，每次迭代计算量不大
缺点：准确度下降，并不是全局最优；不易于并行实现；总体迭代次数比较多
批量梯度下降法:
优点：全局最优解；易于并行实现；总体迭代次数不多
缺点：当样本数目很多时，训练过程会很慢，每次迭代需要耗费大量的时间

参考 : 
    http://blog.csdn.net/yhao2014/article/details/51554910
    https://zhuanlan.zhihu.com/p/22461594
    http://www.cnblogs.com/louyihang-loves-baiyan/p/5136447.html
    http://www.cnblogs.com/louyihang-loves-baiyan/p/5136447.html
"""

# 训练集
# 每个样本点有3个分量 (x0,x1,x2)
x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)]
# y[i] 样本点对应的输出
y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
# 即精度
epsilon = 0.0001

# 学习率
alpha = 0.01
diff = [0, 0]
max_itor = 1000
error1 = 0
error0 = 0
cnt = 0
m = len(x)


# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0

while True:
    cnt += 1 # 尝试次数

    # 参数迭代计算
    for i in range(m):
        # 拟合函数为 y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
        # 计算残差
        diff[0] = (theta0 + theta1 * x[i][1] + theta2 * x[i][2]) - y[i]

        # 梯度 = diff[0] * x[i][j]
        # 下面是推到公式 : theta.j = theta.j-alpha*(损失函数i)*Xi
        # 其中i代表第i个样本,j代表第j个系数项
        # 上面的公式是对每个样本的损失函数求偏导得到的
        # 那么下面的theta系数项的各个值根据上面的公式得到如下:
        theta0 -= alpha * diff[0] * x[i][0]
        theta1 -= alpha * diff[0] * x[i][1]
        theta2 -= alpha * diff[0] * x[i][2]

    # 计算损失函数
    error1 = 0
    for lp in range(len(x)):
        error1 += (y[lp]-(theta0 + theta1 * x[lp][1] + theta2 * x[lp][2]))**2/2 # 损失函数[代价函数]

    if abs(error1-error0) < epsilon: # 如果前后两次损失代价之差的值小于指定的精度,即表明已经靠近最优解的点的位置
        break
    else:
        error0 = error1

    print(' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1))
print('Done: theta0 : %f, theta1 : %f, theta2 : %f' % (theta0, theta1, theta2))
print('迭代次数: %d' % cnt)
