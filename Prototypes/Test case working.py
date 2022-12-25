# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:34:18 2022

@author: user
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from IPython.display import HTML, Image

# These are just the presets I like to use
plt.rc('axes', labelsize=20, titlesize=20)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
from matplotlib import animation
#plt.rcdefaults()

canvas = plt.axes([.25,.525,.45,.45],aspect='equal')
canvas.set_facecolor('midnightblue')
ax2 = plt.axes([.1,.2,.8,.2])

AU = 1.496e11 #m
Msun = 1.989e30 #kg
year = 60*60*24*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = 0.01 #small number to avoid division by 0
N=2
x1, x2 = -.5*AU, .5*AU, #m
y1, y2 = 0, 0 #m
vx1, vx2 = 0, 0 #m/s
vy1, vy2 = -15000, 15000 #m/s

M1, M2 = Msun, Msun #kg
masses = np.array([M1, M2]).reshape(2,1)

x0 = np.array([x1, x2])
y0 = np.array([y1, y2])
vx0 = np.array([vx1, vx2])
vy0 = np.array([vy1, vy2])
  

steps = 512*2
ti = 0
tf = 5*year
time = np.linspace(ti, tf, steps)


init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index

def ode(init, time, masses, G, off, size, d):
    x = init[0:d].reshape(d,1) #converts from list to vector
    y = init[d:d*2].reshape(d,1)
    vx = init[d*2:d*3].reshape(d,1)
    vy = init[d*3:d*4].reshape(d,1)
    #Gives all relative distances
    dx = x.T - x
    dy = y.T - y
    v = np.append(vx, vy)
    
    
    dr3_inv = (dx**2 + dy**2 + off**2)**-(3/2)
    ax = (dx * dr3_inv) @ masses
    ay = (dy * dr3_inv) @ masses
    
    a = G*np.append(ax, ay) #equivalent to the original vx, vy
    return np.append(v, a) #returns 1D array of v, a

sol = odeint(ode, init, time, args=(masses, G, off, size, d))
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T


def T(v, masses):
    return .5 * sum(masses*v**2)

def U(x, y, masses):
    x = x.reshape(steps, d, 1)
    y = y.reshape(steps, d, 1)

    #Gives all relative distances
    dx = np.swapaxes(x,1,2) - x
    dy = np.swapaxes(y,1,2) - y
    dr = np.sqrt(dx**2 + dy**2 + off**2)
    dr_in = 1/dr
    mM = np.triu(-(masses*masses.T)*dr_in,1)
    U = G*np.sum(np.sum(mM, axis = 1), axis = 1)
    return U


   
    
Ttot = T(vx, masses) + T(vy,masses)
Utot = U(x, y, masses)

Etot = Ttot + Utot
dE = (Etot - Etot[0])/Etot[0] 


canvas.plot(x[:,0]/AU, y[:,0]/AU, 'r.', markersize = '1')
canvas.plot(x[:,1]/AU, y[:,1]/AU, 'c.', markersize = '1')


#ax2.plot(time/year, x[:,0]/AU, 'b')
#ax2.plot(time/year, x[:,1]/AU, 'r')


ax2.plot(time/year, dE*1e5, 'k')
ax2.set(ylabel = 'd$E$\n  $[10^{-5}]$')


#__________________________Animation of plot___________________________________


fig, ax = plt.subplots()
ax.axis('equal')
ax.set_facecolor('midnightblue')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))

#initialization
def init():
    return []


#animation function
x = x/AU
y = y/AU
def animate(i):
    #extracts trails
    k = 50 #points in any given trail
    if i < k:
        trails = (x[0:i,0],y[0:i,0], 'r.',
                  x[0:i,1],y[0:i,1], 'c.')
    else:
        trails = (x[i-k:i,0],y[i-k:i,0], 'r.',
                  x[i-k:i,1],y[i-k:i,1], 'c.')
    
    bodies = (x[i,0],y[i,0], 'r',
              x[i,1],y[i,1], 'c')
    
    # * expands list of points
    ax.cla()
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.plot(*trails, markersize='2')
    ax.plot(*bodies, markersize='8', marker = 'o')
    return fig
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=steps, interval=1)
anim