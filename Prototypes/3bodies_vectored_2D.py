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


canvas = plt.axes([.2,.35,.6,.6],aspect='equal')
canvas.set_facecolor('midnightblue')
etot_ax = plt.axes([.1,.075,.8,.2])

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

x = np.array([x1, x2, x3])
y = np.array([y1, y2, y3])
vx = np.array([vx1, vx2, vx3])
vy = np.array([vy1, vy2, vy3])
  

steps = 512
ti = 0
tf = year
time = np.linspace(ti, tf, steps)


init = np.append(np.append(np.append(x, y), vx), vy)
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
x = sol[:,0:d]/AU
y = sol[:,d:2*d]/AU




'''
for t in range(0,steps):
    canvas.cla()
    canvas.set(xlim=(-2.5,2.5), ylim=(-2.5,2.5))
    canvas.plot(x[t,0], y[t,0], 'ro')
    canvas.plot(x[t,1], y[t,1], 'co')
    canvas.plot(x[t,2], y[t,2], color = 'gold', marker = 'o')
    print(t, '/', steps)
    plt.pause(0.0001)
    plt.show()
'''

canvas.plot(x[:,0], y[:,0], 'r.', markersize = '1')
canvas.plot(x[:,1], y[:,1], 'c.', markersize = '1')
canvas.plot(x[:,2], y[:,2], 'y.', markersize = '1')


fig, ax = plt.subplots()

ax.set_xlim((-2.5, 2.5))
ax.set_ylim((-2.5, 2.5))

line1, = ax.plot([], [], 'ro')

# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    return (line1,)


# animation function. This is called sequentially
def animate(i):
    x = sol[i,0:d]/AU
    y = sol[i,d:2*d]/AU
    line1.set_data(x, y)
    return (line1,)

# call the animator. blit=True means only re-draw the parts that 
# have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=512, interval=1, blit=True)
anim












