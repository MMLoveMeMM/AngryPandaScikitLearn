# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:19:58 2017

@author: rd0348
"""

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



