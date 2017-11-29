# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:46:30 2017

@author: rd0348
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import scipy.stats as st
from functools import reduce
import csv

# 创建以为数组
arr1=np.array([1,2,3],dtype=np.int)
print(arr1)

# 创建多为数组
# 创建二维,三维以此类推
twoarr1=np.array([[1,3,2],[3,4,5]])
print(twoarr1)

# 创建9个元素,3*3的数组,并且制定数据类型[非必须]
dularr1=np.arange(9).reshape([3,3])
print(dularr1)

# 创建一个随机数,这个在后面经常用到
rndarr=randn(12).reshape(3,4)
print(rndarr)

# 创建在一个范围内创建指定个数的数组,这个在后面也经常遇见
linsp=np.linspace(1,10,20)
print("创建在指定范围内的数组 : \n",linsp)

# 数组运算,+,*,-,/
yarr = np.arange(9).reshape(3,3)
print("yarr*yarr : \n",yarr*yarr)
print("yarr*yarr : \n",yarr+yarr)
print("yarr-yarr : \n",yarr-yarr)
#print("yarr/yarr : \n",yarr/yarr)

# 数组的转置
tarr=np.arange(6).reshape((2,3))
print("转置之前:\n",tarr)
print("转置之后:\n",tarr.T)

# 数组的属性
print("数组的维数 : \n",yarr.ndim)
print("数组的每一维的大小 : \n",yarr.shape)
print("数组元素个数 : \n",yarr.size)
print("数组元素类型 : \n",yarr.dtype)
print("每个元素所占的字节数 : \n",yarr.itemsize)

# 合并数组
marr=np.ones((2,2))
mbrr=np.eye(2)
print("垂直方向合并 : \n",marr)
print("水平方向合并 : \n",mbrr)

# 数组浅深拷贝
mdrr=mbrr
print("浅拷贝 : \n",mdrr)
mcrr=mbrr.copy()
print("深拷贝 : \n",mcrr)

# 创建矩阵
#创建一个元素全部为0的矩阵
print("矩阵元素均为0 : \n",np.zeros((3,4)))
#创建一个元素全部为1的矩阵
print("矩阵元素均为1 : \n",np.ones((3,4)))
#创建一个对角元素为1的矩阵
print("矩阵对角元素为1 : \n",np.eye(3))

w=np.ones(3)

X=np.ones((5,3))

# 点乘
dotwx=np.dot(X,w) # 标准的矩阵乘法
print("dot : \n",dotwx)
print("dotwx shape : \n",dotwx.shape)
print("X*w : \n",X*w)

# 实现 C=−(y⋅lna^L+(1−y)⋅ln(1−a^L))
def fn(a,y):
    return -(np.dot(y.transpose(),np.log(a))+np.dot((1-y).transpose(),np.log(1-a)))

#如果a=0,y=0代入上面的函数,结果就有问题了,修正如下:
def fnc(a,y):
    return -(np.dot(y.transpose(),np.nan_to_num(np.log(a)))+np.dot((1-y).transpose(),np.nan_to_num(np.log(1-a))))

#加权平均,这个后面应用的非常多,做个图像处理的,对这个会比较熟悉
X = np.array([[.9, .1],
              [.8, .2],
              [.4, .6]])
w=np.array([.2,.2,.6])
print(w.dot(X))
print(np.average(X,axis=0,weights=w))    

# min/max : 返回最小/大值;argmin/argmax : 返回数组最小/大值的索引id
marr=['1','200','993','0']
print("min : \n",min(marr))
print("max : \n",max(marr))
print("argmin : \n",np.argmin(marr)) # 索引从0开始
print("argmax : \n",np.argmax(marr))

# 辨异,降维
x=np.array([[1,2],[3,4]])
print("拷贝降维 : \n",x.flatten())
print("返回视图 : \n",x.ravel())

# 算术运算
# log(exp(x1)+exp(x2))
x1=np.array([1,2,3])
x2=np.array([3,4,5])
print("logexp : \n",np.logaddexp(x1, x2))
# log(x+y)（如果x,y都含有部分指数形式的话）
x=np.array([1,2,3])
y=np.array([3,4,5])
logexp=np.logaddexp(np.log(x), np.log(y))
print("logexp : \n",logexp)

# 高斯公式,这个东西用的就非常的多了
# f(x)=12π−−√σexp(−(x−μ)22σ2)
np.random.normal(loc=0.0, scale=1.0, size=None)
#loc：float 如果没有设置,这个值默认为0
#    此概率分布的均值（对应着整个分布的中心centre）
#scale：float 其实就是公式中的sigma
#    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
#size：int or tuple of ints 采样标本数
#    输出的shape，默认为None，只输出一个值
# np.random.randn(size)所谓标准正态分布（μ=0,σ=1）
mu,sigma=0,.1
s=np.random.normal(loc=mu,scale=sigma,size=1000)
print("高斯分布 : \n",s)
s_fit = np.linspace(s.min(), s.max())
# 同等效果的scipy
plt.plot(s_fit, st.norm(mu, sigma).pdf(s_fit), lw=2, c='r')
plt.show()
#分段函数[刺激函数,这个词在后面见得更多]
plt.plot(x, np.clip(x, -5, 5), 'g', lw=2)
plt.show()

# reduce()函数,这个用的也是超级多
# 显示1+2=3,然后3+3=6;然后6+4=10... ...
# 求出了数量总和21
def func(a,b):
    return a+b
red=reduce(func,np.array([1,2,3,4,5,6]))
print("reduce 函数 : \n",red)
# 也可以增加一个初始值
# 100+1开始,总和121
reds=reduce(func,np.array([1,2,3,4,5,6]),100)
print("reduce 函数 : \n",reds)


# 保存读取数据 这个在训练的时候,经常要保存或者读取数据出来训练
# tofile() 和fromfile() 二进制方式读写
a=np.arange(0,12)
a.shape=(3,4)
a.tofile("a.bin")
b = np.fromfile("a.bin",dtype=np.int)
print("读取二进制文件内容 : \n",b)

# save()和load()
np.save("a.npy",a)
c=np.load("a.npy")
print("读取save保存的数据 : \n",c)

#savez() 保存多个对象
a=np.array([[1,2,3],[4,5,6]])
b=np.arange(0,1.0,0.1)
c=np.sin(b)
np.savez("result.npz",a,b,c_arr=c)
#加载差不多
r=np.load("result.npz")
print("第一个元素 : \n",r["arr_0"])
print("第二个元素 : \n",r["arr_1"])
print("第三个元素 : \n",r["c_arr"])

# savetxt和loadtxt()
a=np.arange(0,12,0.5).reshape(4,-1)
np.savetxt("a.txt",a)
d=np.loadtxt("a.txt")
print("文本方式读写 : \n",d)
np.savetxt("c.txt",a,fmt="%d",delimiter=",")
dp=np.loadtxt("c.txt",delimiter=",")
print("读取时指定分隔符间开元素 : \n",dp)

# 读写对象
a=np.arange(8)
b=np.add.accumulate(a)
c=a+b;
f=open("object.npy","wb")
np.save(f,a)
np.save(f,b)
np.save(f,c)
f.close()
f=open("object.npy","rb")
print("读取第一个对象 : \n",np.load(f))
print("读取第二个对象 : \n",np.load(f))
print("读取第三个对象 : \n",np.load(f))
f.close()

# 下面csv文件读写,这种读取数据的方式,后面也用的比较多
# 更多参考 : https://docs.python.org/2/library/csv.html
datas = [['name', 'age'],
         ['Bob', 14],
         ['Tom', 23],
        ['Jerry', '18']]
with open("csvfile.csv",'w',newline='') as f:
    writer=csv.writer(f)
    for row in datas:
        writer.writerow(row)
# 读csv文件内容
print("读csv文件内容 : ")
with open("csvfile.csv") as f:
    reader=csv.reader(f)
    for row in reader:
        print(reader.line_num,row)










