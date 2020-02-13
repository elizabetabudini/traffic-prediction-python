# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:26:36 2020

@author: S19147099
"""
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wine = pd.read_csv("datasets/wine.csv")

#fig=plt.figure()
#ax = fig.add_subplot(1,1,1)
##Variable
#ax.boxplot(wine['quality'])
#plt.show()
#
#var = wine.groupby('type').quality.mean() #grouped sum of sales at Gender level
#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlabel('Type')
#ax1.set_ylabel('Mean of quality')
#ax1.set_title("Type wise quality mean")
#var.plot(kind='bar')

#quality bar chart
var = wine.groupby('quality').quality.count() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_ylabel("count")
ax1.set_title("Quality frecuency")
var.plot(kind='bar')

#pH bar chart
var = wine.groupby('pH').pH.count() #grouped sum of sales at Gender level
fig = plt.figure()
plt.hist(var, bins=6)
var.plot(kind='bar')



#x = np.linspace(0, 10, 100) 
#y = np.cos(x) 
#z = np.sin(x)

#fig = plt.figure() 
#fig2 = plt.figure(figsize=plt.figaspect(2.0))
#fig.add_axes() 
#ax1 = fig.add_subplot(221) # row-col-num 
#ax3 = fig.add_subplot(212)
#fig3, axes = plt.subplots(nrows=2,ncols=2) 
#fig4, axes2 = plt.subplots(ncols=3)
#  lines = ax.plot(x,y) 
# ax.scatter(x,y)
# axes[0,0].bar([1,2,3],[3,4,5])