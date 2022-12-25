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
Rsun = 6.9634e8
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = AU/10**8 #small number to avoid division by 0
  
N = 5
ti = 0
tf = year
steps = int(tf/day*N*500)
time = np.linspace(ti, tf, steps)


#__________________________Code for N random bodies____________________________


np.random.seed(5)
masses = np.random.uniform(Msun/10, Msun, N).reshape(N,1)  #Variable masses
rads = Rsun*(masses/Msun)**0.8 #Radius's of main sequence stars
drads2 = (rads.T - rads)**2 #squaring because sqrt is harder than squaring and
#not squaring means id need to sqrt the dr's

x0 = np.random.uniform(-AU, AU, N)  #random positions in range +/- 2AU 
y0 = np.random.uniform(-AU, AU, N)
x0 -= np.mean(x0)
y0 -= np.mean(y0) 

vx0 = np.random.uniform(-15000, 15000, N) #random velocities in range +/- 15km/s
vy0 = np.random.uniform(-15000, 15000, N)


#This defines the collision detection
def collision_vel(m1, m2, v1, v2, pos1, pos2):
    M = m1 + m2
    dv, dpos = v1-v2, pos1-pos2
    norm = np.linalg.norm(dpos)**2
    u1 = v1 - 2*m2*np.dot(dv, dpos)*dpos/(norm*M)
    u2 = v2 - 2*m1*np.dot(-dv, -dpos)*-dpos/(norm*M)
    return u1, u2
    

init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index

def ode(init, time, masses, drads2, G, off, size, d):
    m = masses
    x = init[0:d].reshape(d,1) #converts from list to vector
    y = init[d:d*2].reshape(d,1)
    vx = init[d*2:d*3].reshape(d,1)
    vy = init[d*3:d*4].reshape(d,1)
    #Gives all relative distances
    dx = x.T - x
    dy = y.T - y
    dr2 = dx**2 + dy**2 + off**2

    col = dr2 - drads2 #less than 0 means a collision
    v = np.array([vx,vy]).reshape(2,N)
    pos = np.array([x,y]).reshape(2,N)
    col = np.triu(col,1)
    
    #Call the colission detector to work when the col value is < 0 
    for i in range(0,N):
        for j in range(0,N):
            if col[i,j] < 0:  
                v[i], v[j] = collision_vel(m[i],m[j],v[i],v[j],pos[i],pos[j])
    vx, vy = v[0], v[1]
    v = np.append(vx, vy)
    
    
    dr3_inv = (dx**2 + dy**2 + off**2)**-(3/2)
    ax = (dx * dr3_inv) @ masses
    ay = (dy * dr3_inv) @ masses
    
    a = G*np.append(ax, ay) #equivalent to the original vx, vy
    return np.append(v, a) #returns 1D array of v, a

sol = odeint(ode, init, time, args=(masses, rads, G, off, size, d))
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


quality = 300 #lower quality shows quicker animation, higher is better, but
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

x = center(shorten(x,jump)/AU)
y = center(shorten(y,jump)/AU)

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
    k = int(len(x)/5) #points in any given trail
    if i < k:
        trailsx, trailsy = x[0:i], y[0:i]
    else:
        trailsx, trailsy = x[i-k:i], y[i-k:i]
    
    
    # * expands list of points
    trails.set_xdata(trailsx)
    trails.set_ydata(trailsy)
    bodies.set_xdata(x[i,:])
    bodies.set_ydata(y[i,:])
    return fig
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=steps, interval=1)
anim












