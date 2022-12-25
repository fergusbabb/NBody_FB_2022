# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#___________________________________Preamble___________________________________
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

#These are just the presets I like to use
plt.rc('axes', labelsize=20, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#plt.rcdefaults()


#______________________________Initialize figures______________________________
print('Initializing figures...')

fig  = plt.figure(figsize=(13, 6))
fig.tight_layout()
fig2 = plt.figure(figsize=(8,7))

#These parameters were temporarily changed to produce the report figures
widths  = [3, .1, 3]
heights = [2]

gs = fig.add_gridspec(ncols = 3, nrows = 1, width_ratios = widths,
                          height_ratios = heights)


canvas = fig.add_subplot(gs[0, 0], aspect = 'equal')
enerax = fig.add_subplot(gs[0, 2])


animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')

animax.set_facecolor('navy')
canvas.set_facecolor('navy')

#_________________________Define Constants and conditions______________________
print('Defining constants and initial conditions...')

AU = 1.496e11 #m
M = 1.989e30 #kg
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2

tau = np.sqrt((AU**3)/(G*M)) #one unit of time
N = 3
lim = 2 #AU
jump = 1 #number of frames skipped when plotting to speed it up
#for high tf and low steps, jump should be very close to 0
off = AU/1e11 #relatively small number to avoid division by 0

#Change these hash lines to display different solutions

vals = np.loadtxt('Moth_I_vals.txt', delimiter=',')
#vals = np.loadtxt('Butterfly_III_vals.txt', delimiter=',')
#vals = np.loadtxt('S_pineapple_vals.txt', delimiter=',')
#vals = np.loadtxt('Triple_rings_vals.txt', delimiter=',')
#vals = np.loadtxt('Hand_in_hand_in_oval_vals.txt', delimiter=',')
#vals = np.loadtxt('Oval_cat_ship_vals.txt', delimiter=',')
#vals = np.loadtxt('Flower_in_circle_vals.txt', delimiter=',')
#vals = np.loadtxt('Goggles_vals.txt', delimiter=',')
#vals = np.loadtxt('Yin_Yang_III_vals.txt', delimiter=',')



masses = vals[0, 0:N]*M
x0  = vals[1, 0:N]*AU
y0  = vals[2, 0:N]*AU 
vx0 = vals[3, 0:N]*AU/tau
vy0 = vals[4, 0:N]*AU/tau
#Use unit values

period = vals[5, 0] #determine how long to plot
ti = 0
tf = tau*period
steps = int(period*1000) #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
time = np.linspace(ti, tf, steps)
frames = 2**8

init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index


#__________________Main ODE, vectorised________________________________________
print('Solving ODE...')

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

#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8
#Change rtol here to see effect
sol = odeint(ode, init, time, args=(masses,G,off,size,d), rtol=rtol)
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T

#_________________________Energy calculations__________________________________
print('Finding energies...')
#Same block as in validation test.py
def T(v2, masses):
    return np.sum(v2.T*masses, axis = 1)/2

def U(x, y, masses):
    U = np.zeros(steps)
    X, Y = x.copy(), y.copy()
    for t in range(0,steps):
        x = X[t,:].reshape(d,1)
        y = Y[t,:].reshape(d,1)
        dx = x.T - x
        dy = y.T - y
        dr = np.sqrt(dx**2 + dy**2 + off**2)
        term = 0
        #I have no idea why I couldnt vecotise this - they gave completely 
        #different results!
        for i in range(0,N):
            for j in range(0,N):
                if j != i:
                    term += -G*masses[i]*masses[j]/(2*dr[i,j])
        U[t] += term
    return U

Ttot = T(vx**2 + vy**2, masses)
Utot = U(x, y, masses)

#Find relative energy distance
Etot = Ttot + Utot
dE = (Etot - Etot[0])/Etot[0]

enerax.plot(time/tau, dE, 'k')


#______________________________Trim Solution___________________________________

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr)-1, numElems)).astype(int)
    return Arr[idx]

#Evenly sample the full ODE for plotting
x = shorten(x,jump)/AU
y = shorten(y,jump)/AU
x -= np.mean(x)
y -= np.mean(y)
#______________________________Initialize Plots________________________________
print('Initializing plots...')

clrs  = ['chartreuse', 'deeppink', 'darkorange' ] 
    
#Initialize trails
trails, = animax.plot([], [], '.', c = 'lightblue', markersize = '.8')

#Initialize tuple for bodies
aBodies = [] #a for animation bodies
cBodies = [] #c for canvas - ie the static main plot


#Fill Bodies tuple with N lines, one for each body
for j in range(N):
    #Define handles for each body with corresponding size and colour
    abody_j  = animax.plot([], [],  'o', c = clrs[j],
                           markersize = 5)[0]
    canvas_j = canvas.plot([], [],  '.', c = clrs[j], 
                           markersize = .2, lw=.2)[0]
    #Append these handles to their respective tuple
    aBodies.append(abody_j)
    cBodies.append(canvas_j)



#_________________________Define plot parameters_______________________________
h = 1.5
ylim1 = h*np.min(y)  
ylim2 = h*np.max(y) 
xlim1 = h*np.min(x)  
xlim2 = h*np.max(x)  

canvas.set(xlim = (-h, h), ylim = (-h, h),
           xticks = [-h, 0, h], yticks = [-h, 0, h],
           xlabel = '$x$', ylabel = '$y$', title='i.')
animax.set(xlim = (xlim1, xlim2), ylim = (ylim1, ylim2), 
           xlabel = '$x$', ylabel = '$y$')

enerax.set(xlim = (ti, tf/tau), xlabel = '$t$', ylabel = 'd$E$', title='ii.')

animax.set_ylabel(animax.get_ylabel(), rotation = 0)
canvas.set_ylabel(canvas.get_ylabel(), rotation = 0)
enerax.set_ylabel(enerax.get_ylabel(), rotation = 0)
enerax.yaxis.set_label_coords(-.15, .5)

#_____________________________Set Plot Data____________________________________
print('Plotting data...')

#initialization for many bodies (ie more than 1)
def init():
    for body in aBodies:
        body.set_data([],[])
    return body


#loop that fills the cbod_nr-th point for all points in the tuple cBodies
for cbod_nr, body in enumerate(cBodies):
    body.set_data(x[:, cbod_nr], y[:, cbod_nr])


#______________________________Animation Plot__________________________________
print('Animating...')
quality = 500
#To avoid plotting every single timestep (could be 1 million frames +)
fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump2 = int(len(x)/frames)
x = shorten(x,jump2)
y = shorten(y,jump2)

def animate(i):
    #mod to avoid calling over index > frames
    i = i % int(len(x))
    
    k = 50 #points in any given trail
    
    #Plot trails of length k behind the body
    if i < k:
        trailsx, trailsy = x[0:i], y[0:i]
    else:
        trailsx, trailsy = x[i-k:i], y[i-k:i]      
    trails.set_data(trailsx, trailsy) #plots the trails

    
    #Set the animation body data 
    for abod_nr, body in enumerate(aBodies):
        body.set_data(x[i, abod_nr], y[i, abod_nr])

    return fig2


anim = animation.FuncAnimation(fig2, animate, init_func=init,
                               frames = int(frames), interval=1)

#Save animation as gif (not possible from figure GUI) and show
#anim.save('Burraus_test.gif'.format(N), dpi=200, 
#          writer=PillowWriter(fps=40))

print('Finished.')

anim












