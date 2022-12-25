# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 22:09:12 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = 0.7, 200
N = 2000
a = 20

def f(x):
    return (x**-1.5)


x = np.linspace(xmin, xmax, N)
y = f(x)
#plt.plot(x,y, 'r')
xr = np.random.uniform(xmin, xmax, N)
yr = np.random.uniform(0, np.max(y), N)

j = 0
while j < N:
    if yr[j] <= f(xr[j]):
        j += 1
    elif yr[j] > f(xr[j]):
        yr[j] = np.random.uniform(0,np.max(y))
        xr[j] = np.random.uniform(xmin, xmax)


plt.hist(xr)
#plt.plot(x, IMF(x))

