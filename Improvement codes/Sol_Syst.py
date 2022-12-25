# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

#These are just the presets I like to use
plt.rc('axes', labelsize=20, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#plt.rcdefaults()

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


fig  = plt.figure(figsize=(16,8))
fig2 = plt.figure(figsize=(9,9))

widths  = [2, .1,  2]
heights = [2]
gs = fig.add_gridspec(ncols = 3, nrows = 1, width_ratios = widths,
                          height_ratios = heights)

canvas1 = fig.add_subplot(gs[0, 0], aspect='equal')
canvas2 = fig.add_subplot(gs[0, 2], aspect='equal')
canvas1.set_facecolor('midnightblue')
canvas2.set_facecolor('midnightblue')

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
tf = year*65
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
X0 = x0.copy()/AU

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


#__________________Solve and plot as a static image____________________________
#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8 #Smaller -> more accurate
rtol = rtol/1000

sol = odeint(ode, init, time, args = (masses, G, off, size, d), rtol = rtol)

x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T



w = 1.5*x0[4]/AU #AU - set the axis limit, above 5 things start to overlap
w2 = x0[-1]/AU

clrs = ['gold','sandybrown','yellow','lime','red','orange','darkkhaki', 'cyan',
                                    'mediumpurple', 'white']
m_size = ['7', '1', '2', '2', '1.5', '4', '3', '3', '3', '.5']

labels = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
          'Uranus', 'Neptune', 'Pluto']

for p in range(0,5):
    canvas1.plot(x[:,p]/AU, y[:,p]/AU, 
                markersize = m_size[p], color = clrs[p], label=labels[p])
for p in range(0,N):
    canvas2.plot(x[:,p]/AU, y[:,p]/AU, 
                markersize = m_size[p], color = clrs[p], label=labels[p])    


cleg1 = canvas1.legend(facecolor = 'midnightblue', prop={'size': 14})
for legend_text in cleg1.get_texts():
        legend_text.set_color('white')
cleg2 = canvas2.legend(facecolor = 'midnightblue', prop={'size': 14})
for legend_text in cleg2.get_texts():
        legend_text.set_color('white')

canvas1.set(xlabel = '$x$[AU]', ylabel = '$y$[AU]', xticks=[-int(X0[4]), 0,
                                                            int(X0[4])],
            yticks=[-int(X0[4]), 0, int(X0[4])])
canvas2.set(xlabel = '$x$[AU]', ylabel = '$y$[AU]', xticks=[-int(X0[N-1]), 0,
                                                            int(X0[N-1])],
            yticks=[-int(X0[N-1]), 0, int(X0[N-1])])
canvas1.set_ylabel(canvas1.get_ylabel(), rotation = 0)
canvas2.set_ylabel(canvas2.get_ylabel(), rotation = 0)
canvas1.yaxis.set_label_coords(-.1, .55)
canvas2.yaxis.set_label_coords(-.1, .55)

#__________________________Animation of plot___________________________________

quality = 100 #lower quality shows quicker animation, higher is better, but
#too high leads to very slow updates


fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump = int(len(x)/frames)


#To avoid plotting every single timestep (could be 1 million frames +)
def shorten(Arr, jump): 
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr) - 1, numElems)).astype(int)
    return Arr[idx]

x, y = shorten(x,jump)/AU, shorten(y,jump)/AU


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

animax.set_xlim((-int(X0[5]), int(X0[5])))
animax.set_ylim((-int(X0[5]), int(X0[5])))
aleg = animax.legend(facecolor = 'midnightblue', prop={'size': 12})

for legend_text in aleg.get_texts():
        legend_text.set_color('white')
    
#animation function
def animate(i):
    #Use mod to avoid extension of indices ie back to start when time ended
    i = i % int(len(x))
    
    #k = int(len(x)/5) #points in any given trail
    k = 50
    
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

#anim.save('Solar_system.gif', dpi=100, writer=PillowWriter(fps=40))

anim








