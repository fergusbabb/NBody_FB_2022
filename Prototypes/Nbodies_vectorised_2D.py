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

seed = 4
np.random.seed(seed)

#_____________________Set up axes and define constants_________________________

canvas = plt.axes([.2,.35,.6,.6],aspect='equal')
#canvas.set_facecolor('midnightblue')
etot_ax = plt.axes([.1,.075,.8,.2])

AU = 1.496e11 #m
Msun = 1.989e30 #kg
year = 60*60*24*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = 0.01 #small number to avoid division by 0

steps = int(1024/4)
ti = 0
tf = year
time = np.linspace(ti, tf, steps)

#__________________________Code for N random bodies____________________________

N = 4

masses = np.random.uniform(Msun, Msun, N)  #Variable masses

x0 = np.random.uniform(-2*AU,2*AU, N)  #randomly positions in range +/- 2AU 
y0 = np.random.uniform(-2*AU,2*AU, N)   

vx0 = np.random.uniform(-15000,15000, N) #random velocities in range +/- 15km/s
vy0 = np.random.uniform(-15000,15000, N)



init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index


#___________________________Definition of ODE__________________________________


def ode(init, time, masses, G, off, size, d):
    x = init[0:d].reshape(d,1) #converts from list to vector
    y = init[d:d*2].reshape(d,1)
    
    #Gives all relative distances
    dx = x.T - x
    dy = y.T - y    
    
    dr3_inv = (dx**2 + dy**2 + off**2)**-(3/2)
    
    ax = (dx * dr3_inv) @ masses
    ay = (dy * dr3_inv) @ masses
    
    a = G*np.append(ax, ay) #equivalent to the original vx, vy

    return np.append(init[d*2:size], a) #returns 1D array of v, a


#____________________________________Plots_____________________________________


sol = odeint(ode, init, time, args=(masses, G, off, size, d))
x = sol[:,0:d]/AU
y = sol[:,d:2*d]/AU
#Determine plotting center
xmean = np.mean(x)
ymean = np.mean(y)

#Plots for 3 bodies
#canvas.plot(sol[:,0]/AU, sol[:,3]/AU, 'r.', markersize = '2')
#canvas.plot(sol[:,1]/AU, sol[:,4]/AU, 'c.', markersize = '2')
#canvas.plot(sol[:,2]/AU, sol[:,5]/AU, 'y.', markersize = '2')

#Plot for N bodies
canvas.plot(x, y)


for t in range(0,steps):
    canvas.cla()
    #canvas.plot(pos[:,0]/au*100,pos[:,1]/au*100, 'bo')
    canvas.set(xlim=(-2.5,2.5), ylim=(-2.5,2.5))
    canvas.plot(x[t],y[t], 'o')
    plt.pause(0.001)
    plt.show()


canvas.plot(x, y, alpha = 0.75)

