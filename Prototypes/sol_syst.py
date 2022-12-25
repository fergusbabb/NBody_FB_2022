# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

# These are just the presets I like to use
plt.rc('axes', labelsize=20, titlesize=20)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})


from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


fig  = plt.figure(figsize=(9,9))
fig2 = plt.figure(figsize=(9,9))

canvas = fig.add_axes([.1,.1,.8,.8], aspect='equal')
canvas.set_facecolor('midnightblue')

animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')
animax.set_facecolor('midnightblue')


#_______________________________Define Constants_______________________________
AU = 1.496e11 #m
Msun = 1.989e30 #kg
Rsun = 6.9634e8
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
off = AU/100 #small number to avoid division by 0
c = 299792458 #m/s

N = 9 #increase to 10 to make pluto happy, I reccomend staying with 9 as larger
#makes the animation plot points overlap to see pluto, (Still wanted pluto!)

ti = 0
tf = year*10
steps = int(tf/day*N*50/10)
time = np.linspace(ti, tf, steps)

seed = random.randrange(1000)
rng = random.Random(seed)
print("Seed was:", seed) #for repeatablility

#______________________________Initial conditions______________________________

vals  = np.loadtxt('solarsys_vals.txt', delimiter = ',')
masses = vals[0,0:N]
x0     = vals[1,0:N]
y0     = vals[2,0:N]
vx0    = vals[3,0:N]
vy0    = vals[4,0:N]


angle = np.random.uniform(0, 2*np.pi, N)


vx0, vy0 = -vy0*np.sin(angle), vy0*np.cos(angle)
x0, y0  =  x0*np.cos(angle), x0*np.sin(angle)



#________________Append initial states to array for odeint_____________________

init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d    = int(size/4) #defines smallest needed index



#__________________Main ODE, vectorised________________________________________

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


#_________________________Energy calculations__________________________________
'''
#function for kinetic energy
def T(v, masses):
    return .5*sum(masses*v**2)
'''

#function for potential energy
def U(x, y, masses):
    x = x.reshape(steps, d, 1)
    y = y.reshape(steps, d, 1)

    #Gives all relative distances similar to in ode but we have 3D array now
    #so using swap axes instead
    
    dx = np.swapaxes(x,1,2) - x
    dy = np.swapaxes(y,1,2) - y
    dr = np.sqrt(dx**2 + dy**2 + off**2)
    dr_in = 1/dr
    
    #We only care about the upper right triange, which is found with triu
    mM = np.triu(-(masses*masses.T)*dr_in,1)
    #Potential
    U = G*np.sum(np.sum(mM, axis = 1), axis = 1)
    return U

'''
Same comments 
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

#Find relative energy distance
E0 = T0 + U0
Etot = Ttot + Utot
dE = (Etot - E0)/E0

#ax3 = plt.axes([.2,.05,.6,.1])
#ax3.plot(time/year, dT, 'b')
#ax3.plot(time/year, dU, 'r')
#ax3.plot(time/year, dE, 'k')
'''

#__________________Solve and plot as a static image____________________________
#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8 #Smaller -> more accurate
rtol = rtol/1000

sol = odeint(ode, init, time, args = (masses, G, off, size, d), rtol = rtol)

x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T

w = x0[6]/AU #AU - set the axis limit, above 5 things start to overlap

clrs = ['gold','sandybrown','yellow','lime','red','orange','darkkhaki', 'cyan',
                                    'mediumpurple', 'white']
m_size = ['7', '1', '2', '2', '1.5', '4', '3', '3', '3', '.5']

labels = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
          'Uranus', 'Neptune', 'Pluto']

for p in range(0,N):
    canvas.plot(x[:,p]/AU, y[:,p]/AU, 
                markersize = m_size[p], color = clrs[p], label=labels[p])
    



canvas.set_xlim((-w,w)) #index -> inital radius of that indexed planet
canvas.set_ylim((-w,w))
cleg = canvas.legend(facecolor = 'midnightblue', prop={'size': 14})
for legend_text in cleg.get_texts():
        legend_text.set_color('lightblue')

#__________________________Animation of plot___________________________________

quality = 1000 #lower quality shows quicker animation, higher is better, but
#too high leads to very slow updates


fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump = int(len(x)/frames)


#To avoid plotting every single timestep (could be 1 million frames +)
def shorten(Arr, jump): 
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr) - 1, numElems)).astype(int)
    return Arr[idx]


''' #Unnecessary, I prefer without
def center(Arr):
    for i in range(N):
        Arr[:,i] -= Arr[:,0] #centers to make sun static
    return Arr'''

x, y = shorten(x,jump)/AU, shorten(y,jump)/AU
#x, y = center(x), center(y)


#______________________________Initialise plots________________________________

trails, = animax.plot([], [], '.', c ='lightblue', markersize = '.2', lw='.2')

#Initialize tuple for bodies
aBodies = []

#Fill Bodies tuple with N lines, one for each body
for j in range(N):
    body_j = animax.plot([],[], 'o', color=clrs[j], markersize = m_size[j],
                     label=labels[j])[0]
    aBodies.append(body_j)
    
#initialization for many bodies (ie more than 1)
def init():
    for body in aBodies:
        body.set_data([],[])
    return body

animax.set_xlim((-w, w))
animax.set_ylim((-w, w))
aleg = animax.legend(facecolor = 'midnightblue', prop={'size': 12})
for legend_text in aleg.get_texts():
        legend_text.set_color('lightblue')
    
#animation function
def animate(i):
    #Use mod to avoid extension of indices ie back to start when time ended
    i = i % int(len(x))
    
    #k = int(len(x)/5) #points in any given trail
    k = 100
    
    #Plot trails of length k behind the body
    if i < k:
        trailsx, trailsy = x[0:i], y[0:i]
    else:
        trailsx, trailsy = x[i-k:i], y[i-k:i]
        
    trails.set_data(trailsx, trailsy) #plots the trails
    

    #Set the body data, while allowing for different colours and sizes 
    for bod_nr, body in enumerate(aBodies):
        body.set_data(x[i,bod_nr], y[i,bod_nr])

    return fig2
    
anim = FuncAnimation(fig2, animate, init_func=init,
                               frames=int(frames), interval=1)

#anim.save('Solar_system.gif', dpi=300, writer=PillowWriter(fps=20))

anim








