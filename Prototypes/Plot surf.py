# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:56:06 2022

@author: user
"""

#___________________________________Preamble___________________________________
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#These are just the presets I like to use
plt.rc('axes', labelsize=16, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

#plt.rcdefaults()



#______________________________Initialize figures______________________________

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


lim = 2
N = 5
G = 1
pts = 1024
x = np.linspace(-lim, lim, pts)
y = np.copy(x)

a = np.random.uniform(-lim, lim, N)
b = np.random.uniform(-lim, lim, N)
m = np.random.uniform(.1, 2, N)
x, y = np.meshgrid(x, y)

def Usurf(a, b):
    return -G/(np.sqrt((x-a)**2 + (y-b)**2))

U = np.zeros(shape=(pts,pts))
for i in range(0, N):
    U += m[i]*Usurf(a[i], b[i])
    

zmin = -5*lim

for i in range(len(x)):
    for j in range(len(y)):
        if U[i, j] <= zmin:
            U[i,j] = zmin



surf = ax.plot_surface(x, y, U, cmap=cm.jet)
ax.set_zlim(zmin, 0)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)





