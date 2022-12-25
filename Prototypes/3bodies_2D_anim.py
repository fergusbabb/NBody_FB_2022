# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

plt.rcParams['animation.html'] = 'html5'
from matplotlib import animation


canvas = plt.axes([.25,.525,.45,.45],aspect='equal')
canvas.set_facecolor('midnightblue')
ax2 = plt.axes([.1,.275,.8,.2])

AU = 1.496e11 #m
Msun = 1.989e30 #kg
year = 60*60*24*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = 0.01 #small number to avoid division by 0

x1, x2, x3 = -.5*AU, .5*AU, .1*AU #m
y1, y2, y3 = 0, 0, .1*AU #m
vx1, vx2, vx3 = 0, 0, 0 #m/s
vy1, vy2, vy3 = -15000, 15000, 0 #m/s

M1, M2, M3 = Msun, Msun, Msun #kg
masses = np.array([M1, M2, M3]).reshape(3,1)

x0 = np.array([x1, x2, x3])
y0 = np.array([y1, y2, y3])
vx0 = np.array([vx1, vx2, vx3])
vy0 = np.array([vy1, vy2, vy3])
  
N = 3
steps = 512*4
ti = 0
tf = year
time = np.linspace(ti, tf, steps)


init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index


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

sol = odeint(ode, init, time, args=(masses, G, off, size, d))
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T

def T(v, masses):
    return .5*sum(masses*v**2)

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

def U0func(x, y, masses):
    x = x.reshape(N, 1)
    y = y.reshape(N, 1)
    #Gives all relative distances
    dx = np.swapaxes(x, 0,1) - x
    dy = np.swapaxes(x, 0,1) - y
    dr = np.sqrt(dx**2 + dy**2 + off**2)
    dr_in = 1/dr
    mM = np.triu(-(masses*masses.T)*dr_in,1)
    U = G*np.sum(np.sum(mM, axis = 1), axis = 0)
    return U    
    
Ttot = T(vx, masses) + T(vy,masses)
T0 = sum(T(vx0, masses) + T(vy0, masses))
dT = (Ttot - T0)/T0 

Utot = U(x, y, masses)
U0 = U0func(x0, y0, masses)
dU = (Utot - U0)/U0

E0 = T0 + U0
Etot = Ttot + Utot
dE = (Etot - E0)/E0


canvas.plot(x[:,0]/AU, y[:,0]/AU, 'r.', markersize = '1')
canvas.plot(x[:,1]/AU, y[:,1]/AU, 'c.', markersize = '1')
canvas.plot(x[:,2]/AU, y[:,2]/AU, 'gold', '.', markersize = '1')

ax2.plot(time/year, x[:,0]/AU, 'r')
ax2.plot(time/year, x[:,1]/AU, 'c')
ax2.plot(time/year, x[:,2]/AU, 'gold')

#ax3 = plt.axes([.2,.05,.6,.1])
#ax3.plot(time/year, dT, 'b')
#ax3.plot(time/year, dU, 'r')
#ax3.plot(time/year, dE, 'k')


#__________________________Animation of plot___________________________________


fig, ax = plt.subplots()
ax.axis('equal')
ax.set_facecolor('midnightblue')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))

#initialization
def init():
    return []

colours = ['r', 'c', 'gold']
#animation function
def animate(i):
    #extracts trails
    k = 50 #points in any given trail
    if i < k:
        trails = (x[0:i,0]/AU, y[0:i,0]/AU, 'r',
                  x[0:i,1]/AU, y[0:i,1]/AU, 'c',
                  x[0:i,2]/AU, y[0:i,2]/AU, 'gold')
    else:
        trails = (x[i-k:i,0]/AU, y[i-k:i,0]/AU, 'r',
                  x[i-k:i,1]/AU, y[i-k:i,1]/AU, 'c',
                  x[i-k:i,2]/AU, y[i-k:i,2]/AU, 'gold')
    
    bodies = (x[i,0]/AU, y[i,0]/AU, 'r',
              x[i,1]/AU, y[i,1]/AU, 'c',
              x[i,2]/AU, y[i,2]/AU, 'gold')
    
    # * expands list of points
    ax.cla()
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.plot(*trails, markersize='2', marker = '.')
    ax.plot(*bodies, markersize='8', marker = 'o')
    return fig
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=steps, interval=1)
anim












