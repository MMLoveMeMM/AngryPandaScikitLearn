# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:40:37 2017

@author: rd0348
"""
# 数据源:
# 搜索英国政府数据或美国政府数据来获取数据源。当然，Kaggle是另一个好用的数据源
# 英国政府数据 : https://data.gov.uk/
# 美国政府数据 : https://www.data.gov/
# Kaggle数据 : https://www.kaggle.com/
# 完整的pandas使用参考 : https://www.cnblogs.com/chaosimple/p/4153083.html
import pandas as pd

df=pd.read_csv('raingauges-2014-06.csv',header=0)
df.head(5)
print("读取最前面的五行数据 : \n",df.head(5))
df.tail(5)
print("读取最后面的五行数据 : \n",df.tail(5))

# 这个文件的列名太长,可以调整如下
df.columns=['TIMESTAMP','AllertonBywater','Middleton','Otley','Shadwell','PotteryField']
print("修改列名称后 : \n",df.head(5))

# 数据总条数
print("数据记录总条数 : ",len(df))

# 描述
print("数据描述 : \n",df.describe())

# 过滤字段,eg : 下面只显示Otley字段对应的数据
print("过滤字段<1> : \n",df['Otley'])

# 过滤字段,也可以采用点属性
print("过滤字段<2> : \n",df.Otley)

# 条件过滤
print("条件过滤 : \n",df.Otley>0)

print("条件过滤 : \n",df[(df.Otley>0)&(df.Otley<1)])

# 如果存在行索引
print("如果存在行索引 : \n",df.iloc[10])

# 用字段做索引
df.set_index(['TIMESTAMP'])
print("用字段做索引 : \n",df.head(5))

# 字符串索引
#df.loc['02-Jun-14']
#print("字符串索引 : \n",df.head(5))

# 其他的还有一些数据表格合并,删除,排序,获取标量值之类的操作.

# 绘图,这个绘图方式比Matplotlib要简单
df.plot(x='TIMESTAMP',y=['Middleton','Otley'])

# 操作了上面以后,进行保存
df.to_csv('rain_pandas.csv')