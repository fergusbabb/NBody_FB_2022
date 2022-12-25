# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from IPython.display import HTML, Image
from mpl_toolkits.mplot3d import Axes3D

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
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = AU/10**8 #small number to avoid division by 0
  
N = 3
ti = 0
tf = 4*year
steps = int(tf/day*N*1000)
time = np.linspace(ti, tf, steps)


#__________________________Code for N random bodies____________________________


np.random.seed(5)
masses = np.random.uniform(Msun, Msun, N).reshape(N,1)  #Variable masses
x0 = np.random.uniform(-AU,AU, N)  #randomly positions in range +/- 2AU 
y0 = np.random.uniform(-AU,AU, N)
z0 = np.random.uniform(-AU,AU, N)
x0 -= np.mean(x0)
y0 -= np.mean(y0)
z0 -= np.mean(z0)

vx0 = np.random.uniform(-15000,15000, N) #random velocities in range +/- 15km/s
vy0 = np.random.uniform(-15000,15000, N)
vz0 = np.random.uniform(-15000,15000, N)

posns = np.append(np.append(x0, y0), z0)
vels = np.append(np.append(vx0, vy0), vz0)
init = np.append(posns,vels)
size = int(len(init))
d = int(size/6) #defines smallest needed index

def ode(init, time, masses, G, off, size, d):
    x = init[0:d].reshape(d,1) #converts from list to vector
    y = init[d:d*2].reshape(d,1)
    z = init[d*2:d*3].reshape(d,1)
    
    #Gives all relative distances
    dx = x.T - x
    dy = y.T - y  
    dz = z.T - z  
    
    dr3_inv = (dx**2 + dy**2 + dz**2 + off**2)**-(3/2)
    
    ax = (dx * dr3_inv) @ masses
    ay = (dy * dr3_inv) @ masses
    az = (dz * dr3_inv) @ masses
    
    a = G*np.append(np.append(ax, ay), az) #equivalent to the original vx, vy

    return np.append(init[d*3:size], a) #returns 1D array of v, a

sol = odeint(ode, init, time, args=(masses, G, off, size, d))
x  = sol[:,  0:d]
y  = sol[:,  d:2*d]
z  = sol[:,d*2:3*d]
vx = sol[:,3*d:4*d].T
vy = sol[:,4*d:5*d].T
vz = sol[:,5*d:6*d].T

def T(v, masses):
    return .5*sum(masses*v**2)

def U(x, y, z, masses):
    x = x.reshape(steps, d, 1)
    y = y.reshape(steps, d, 1)
    z = z.reshape(steps, d, 1)

    #Gives all relative distances
    dx = np.swapaxes(x,1,2) - x
    dy = np.swapaxes(y,1,2) - y
    dz = np.swapaxes(z,1,2) - z
    
    dr = np.sqrt(dx**2 + dy**2 + dz**2 + off**2)
    dr_in = 1/dr
    mM = np.triu(-(masses*masses.T)*dr_in,1)
    U = G*np.sum(np.sum(mM, axis = 1), axis = 1)
    return U

def U0func(x, y, z, masses):
    x = x.reshape(N, 1)
    y = y.reshape(N, 1)
    z = z.reshape(N, 1)
    
    #Gives all relative distances
    dx = np.swapaxes(x, 0,1) - x
    dy = np.swapaxes(x, 0,1) - y
    dz = np.swapaxes(z, 0,1) - z
    
    dr = np.sqrt(dx**2 + dy**2 + dz**2 + off**2)
    dr_in = 1/dr
    mM = np.triu(-(masses*masses.T)*dr_in,1)
    U = G*np.sum(np.sum(mM, axis = 1), axis = 0)
    return U    
    
Ttot = T(vx, masses) + T(vy,masses) + T(vz, masses)
T0 = sum(T(vx0, masses) + T(vy0, masses) + T(vz0, masses))
dT = (Ttot - T0)/T0 

Utot = U(x, y, z, masses)
U0 = U0func(x0, y0, z0, masses)
dU = (Utot - U0)/U0

E0 = T0 + U0
Etot = Ttot + Utot
dE = (Etot - E0)/E0


canvas.plot(x[:]/AU, y[:]/AU, markersize = '1')


ax2.plot(time/year, x[:]/AU)

#ax3 = plt.axes([.2,.05,.6,.1])
#ax3.plot(time/year, dT, 'b')
#ax3.plot(time/year, dU, 'r')
#ax3.plot(time/year, dE, 'k')


#__________________________Animation of plot___________________________________


fig, ax = plt.subplots()
ax.axis('equal')
ax = plt.axes(projection='3d')
#ax.set_facecolor('midnightblue')
#ax.set_xlim((-1, 1))
#ax.set_ylim((-1, 1))


quality = 400 #lower quality shows quicker animation, higher is better, but
#too high is hard to see the movement

fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump = int(len(x)/frames)

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    i = np.round(np.linspace(0, len(Arr) - 1, numElems)).astype(int)
    return Arr[i]

def center(Arr):
    for i in range(len(Arr)):
        Arr[i,:] -= np.mean(Arr[i,:])
    return Arr

x = center(shorten(x,jump)/AU)
y = center(shorten(y,jump)/AU)
z = center(shorten(z,jump)/AU)



#initialization
def init():
    return []


trails, = ax.plot3D(x[0,:], y[0,:], z[0,:], '.', markersize='1')
bodies, = ax.plot3D(x[0,:], y[0,:], z[0,:], 'o', markersize='5')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_zlim(-2.2, 2.2)

#animation function
def animate(i):
    #extracts trails
    i = i % int(len(x))
    k = int(len(x)/5) #points in any given trail
    if i < k:
        trailsx, trailsy, trailsz = x[0:i],   y[0:i],   z[0:i]
    else:
        trailsx, trailsy, trailsz = x[i-k:i], y[i-k:i], z[i-k:i]
        
    # * expands list of points
    trails.set_xdata(trailsx)
    trails.set_ydata(trailsy)
    trails.set_3d_properties(trailsz)
    #trails.set_zdata(trailsz)
    
    bodies.set_xdata(x[i,:])
    bodies.set_ydata(y[i,:])
    #bodies.set_3d_properties(z[i,:])
    bodies.set_zdata(z[i,:])
    return 
    
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=steps, interval=1)
#anim


for i in range(steps):
    ax.cla()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-2.2, 2.2)
    ax.plot3D(x[i,:], y[i,:], z[i,:], 'o', markersize='5')
    plt.pause(.001)









