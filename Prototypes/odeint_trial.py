# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# These are just the presets I like to use
plt.rc('axes', labelsize=20, titlesize=20)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})


canvas = plt.axes([.2,.35,.6,.6],aspect='equal')
canvas.set_facecolor('midnightblue')
etot_ax = plt.axes([.1,.075,.8,.2])

AU = 1.496e11 #m
Msun = 1.989e30 #kg
year = 60*60*24*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2

x1, x2 = -.5*AU, .5*AU #m
y1, y2 = 0, 0 #m
vx1, vx2 = 0, 0 #m/s
vy1, vy2 = -15000, 15000 #m/s

M1, M2 = Msun, Msun #kg

steps = 1024
ti = 0
tf = 5*year
time = np.linspace(ti, tf, steps)



x = np.array([x1, x2])
y = np.array([y1, y2])
vx = np.array([vx1, vx2])
vy = np.array([vy1, vy2])

init = np.append(np.append(np.append(x, y), vx), vy)


def ode(init, time, M1, M2, G):
    size = int(len(init))
    d = int(size/4) #defines smallest needed index
    x = init[0:d].reshape(d,1) #converts from list to vector
    y = init[d:d*2].reshape(d,1)

    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dr3_inv = (dx**2 + dy**2)**-(3/2)
    
    ax1 = G*M2*dx*dr3_inv
    ay1 = G*M2*dy*dr3_inv
    
    ax2 = -G*M1*dx*dr3_inv
    ay2 = -G*M1*dy*dr3_inv
    
    a = np.array([ax1, ax2, ay1, ay2]) #equivalent of the original vx, vy

    return np.append(init[d*2:size], a) #returns 1D array of v, a

sol = odeint(ode, init, time, args=(M1, M2, G))

canvas.plot(sol[:,0]/AU, sol[:,2]/AU, 'r.', markersize = '2')
canvas.plot(sol[:,1]/AU, sol[:,3]/AU, 'c.', markersize = '2')

