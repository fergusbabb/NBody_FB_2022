# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 21:23:16 2022

@author: user
"""

import  numpy as np
import matplotlib.pyplot as plt
  

N=9
  
x = np.linspace(0,100,N)
y = np.linspace(0,20,N)
t = np.linspace(0,10,100)


# empty list, will hold color value
# corresponding to x
col =[]

colours = ['gold','brown','yellow','lime','red','orange','cyan','lightblue','purple']


for i in range(len(x)):
      
    # plotting the corresponding x with y 
    # and respective color
    plt.scatter(x[i], y[i], c = colours[i], s = 10,
                linewidth = 0)
'''    
colormap = plt.get_cmap('hsv')
l = np.linspace(0,1,N)
colours2 = []
for k in range(len(l)):
    colours2.append(colormap(l[k]))
    plt.scatter(x[k],y[k],color=colormap(l[k]))
'''
