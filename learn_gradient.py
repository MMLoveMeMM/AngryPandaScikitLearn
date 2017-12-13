# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:38:47 2017
learn_gradient.py
@author: rd0348
给出三种梯度下降算法,用于对比参照
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

"""
下面的三种公式都差不多,但是在选取做梯度的两个样本的方式是不一样的.
<1> : 梯度下降 : 前后两个都是在所有的样本中随机获取,然后进行loss计算,只要loss低于eps即停止;
<2> : 随机梯度下降 : 选择一个预定的点,然后另外一个随机产生(在所有的样本中随机产生),计算loss,loss低于eps即停止;
<3> : 批量梯度下降 : 第一个预定点随机产生以后,第二个点必须是第一个点附近的点
"""
import random
print("梯度下降 : ")
#This is a sample to simulate a function y = theta1*x1 + theta2*x2
input_x = [[1,4], [2,5], [5,1], [4,2]]  
y = [19,26,19,20]  
theta = [1,1]
loss = 10
step_size = 0.001
eps =0.0001 # 结果精度,低于此值,即认为获取最优解
max_iters = 10000
error =0
iter_count = 0
while( loss > eps and iter_count < max_iters):
    loss = 0
    #这里更新权重的时候所有的样本点都用上了
    for i in range (3):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
        theta[0] = theta[0] - step_size * (pred_y - y[i]) * input_x[i][0]
        theta[1] = theta[1] - step_size * (pred_y - y[i]) * input_x[i][1]
    for i in range (3):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
        error = 0.5*(pred_y - y[i])**2
        loss = loss + error
    iter_count += 1
    #print('iters_count', iter_count,'\ttheta: ',theta)

print('最优系数theta: ',theta )
print('final loss: ', loss)
print('梯度次数iters: ', iter_count)

print("-"*60)
print("随机梯度下降 : ")
#随机梯度下降
#每次选取一个随机值，随机一个点更新θ
#This is a sample to simulate a function y = theta1*x1 + theta2*x2
input_x = [[1,4], [2,5], [5,1], [4,2]]  
y = [19,26,19,20]  
theta = [1,1]
loss = 10
step_size = 0.001
eps =0.0001
max_iters = 10000
error =0
iter_count = 0
while( loss > eps and iter_count < max_iters):
    loss = 0
    #每一次选取随机的一个点进行权重的更新
    i = random.randint(0,3)
    pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
    theta[0] = theta[0] - step_size * (pred_y - y[i]) * input_x[i][0]
    theta[1] = theta[1] - step_size * (pred_y - y[i]) * input_x[i][1]
    for i in range (3):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
        error = 0.5*(pred_y - y[i])**2
        loss = loss + error
    iter_count += 1
    #print('iters_count', iter_count,'\ttheta: ',theta)

print('theta: ',theta) 
print('final loss: ', loss)
print('iters: ', iter_count)

print("-"*60)
print("批量随机梯度下降 : ")
# 批量随机梯度下降
# 这里用了2个样本点
#This is a sample to simulate a function y = theta1*x1 + theta2*x2
input_x = [[1,4], [2,5], [5,1], [4,2]]  
y = [19,26,19,20]  
theta = [1,1]
loss = 10
step_size = 0.001
eps =0.0001
max_iters = 10000
error =0
iter_count = 0
while( loss > eps and iter_count < max_iters):
    loss = 0
    
    i = random.randint(0,3) #注意这里，我这里批量每次选取的是2个样本点做更新，另一个点是随机点+1的相邻点
    j = (i+1)%4
    pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
    theta[0] = theta[0] - step_size * (pred_y - y[i]) * input_x[i][0]
    theta[1] = theta[1] - step_size * (pred_y - y[i]) * input_x[i][1]

    pred_y = theta[0]*input_x[j][0]+theta[1]*input_x[j][1]
    theta[0] = theta[0] - step_size * (pred_y - y[j]) * input_x[j][0]
    theta[1] = theta[1] - step_size * (pred_y - y[j]) * input_x[j][1]
    for i in range (3):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]
        error = 0.5*(pred_y - y[i])**2
        loss = loss + error
    iter_count += 1
    #print('iters_count', iter_count,'\ttheta: ',theta)

print('theta: ',theta)
print('final loss: ', loss)
print('iters: ', iter_count)