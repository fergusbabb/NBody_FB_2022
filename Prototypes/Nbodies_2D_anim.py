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


canvas = plt.axes([.25,.525,.45,.45], aspect='equal')
canvas.set_facecolor('midnightblue')
ax2 = plt.axes([.1,.275,.8,.2])

AU = 1.496e11 #m
Msun = 1.989e30 #kg
Rsun = 6.9634e8
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = AU/100 #small number to avoid division by 0
c = 299792458 #m/s
Mmin = Msun/5
Mmax = Msun*150

N = 5
ti = 0
tf = year
steps = int(tf/day*N*750)
time = np.linspace(ti, tf, steps)
BHmass = Msun*1e3

#__________________________Code for N random bodies____________________________

#Define masses according to the distribution that star freq \propto mass^-2

np.random.seed(5)
p = np.linspace(.1,1,N)
p = p**-2/10
masses = (np.random.uniform(Mmin,Mmax,N)*p).reshape(N,1) #Variable masses
masses[0] = BHmass

rads = Rsun*(masses/Msun)**0.8 #Radius's of main sequence stars
BH_rs = 2*G*BHmass/c**2
rads[0] = BH_rs


x0 = np.random.uniform(-AU, AU, N)  #random positions in range +/- 2AU 
y0 = np.random.uniform(-AU, AU, N)
x0[0], y0[0] = 0, 0
x0 -= np.mean(x0)
y0 -= np.mean(y0) 

vx0 = np.random.uniform(-15000, 15000, N) #random velocities in range +/- 15km/s
vy0 = np.random.uniform(-15000, 15000, N)
vx0[0], vy0[0] = 0, 0

    

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

'''
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
'''

canvas.plot(x[:]/AU, y[:]/AU, markersize = '1')
canvas.set_xlim((-2.2, 2.2))
canvas.set_ylim((-2.2, 2.2))
ax2.set_xlim((ti/year, tf/year))
ax2.set_ylim((-2.2, 2.2))
ax2.plot(time/year, x[:]/AU)

#ax3 = plt.axes([.2,.05,.6,.1])
#ax3.plot(time/year, dT, 'b')
#ax3.plot(time/year, dU, 'r')
#ax3.plot(time/year, dE, 'k')


#__________________________Animation of plot___________________________________


fig, ax = plt.subplots()
ax.axis('equal')
ax.set_facecolor('midnightblue')
#ax.set_xlim((-1, 1))
#ax.set_ylim((-1, 1))


quality = 1000 #lower quality shows quicker animation, higher is better, but
#too high leads to very slow updates

fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump = int(len(x)/frames)

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr) - 1, numElems)).astype(int)
    return Arr[idx]

def center(Arr):
    for i in range(len(Arr)):
        Arr[i,:] -= np.mean(Arr[i,:])
    return Arr

x = shorten(x,jump)/AU
y = shorten(y,jump)/AU

#Center to BH
x-=x[0,:]
y-=y[0,:]

#initialization
def init():
    return []


trails, = ax.plot(x[0,0], y[0,0], '.', markersize='1')
bodies, = ax.plot(x[0,0], y[0,0], 'o', markersize='5')
ax.set_xlim((-2.2, 2.2))
ax.set_ylim((-2.2, 2.2))

#animation function
def animate(i):
    #extracts trails
    i = i % int(len(x))
    #k = int(len(x)/5) #points in any given trail
    k = 50
    if i < k:
        trailsx, trailsy = x[0:i], y[0:i]
    else:
        trailsx, trailsy = x[i-k:i], y[i-k:i]
    
    #ax.plot(0,0, 'o', color = 'white')
    # * expands list of points
    trails.set_xdata(trailsx)
    trails.set_ydata(trailsy)
    bodies.set_xdata(x[i,:])
    bodies.set_ydata(y[i,:])
    return fig
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=steps, interval=1)
anim












