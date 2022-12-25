# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#___________________________________Preamble___________________________________
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#from IPython.display import HTML, Image
from matplotlib.colors import LinearSegmentedColormap
#from matplotlib.ticker import LogFormatter 
import random
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import LogNorm

#These are just the presets I like to use
plt.rc('axes', labelsize=16, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

#plt.rcdefaults()



#______________________________Initialize figures______________________________

fig  = plt.figure(figsize=(15,10))
fig2 = plt.figure(figsize=(8,7))

widths  = [2, .1,  2]
heights = [2,  .05, 1]
gs = fig.add_gridspec(ncols = 3, nrows = 3, width_ratios = widths,
                          height_ratios = heights)


canvas = fig.add_subplot(gs[0, 0], aspect='equal')
enerax = fig.add_subplot(gs[0, 2])
xposax = fig.add_subplot(gs[2, :])

animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')

canvas.set_facecolor('navy')
animax.set_facecolor('navy')
xposax.set_facecolor('navy')





#_________________________________Define Constants_____________________________

AU = 1.496e11 #m
Msun = 1.989e30 #kg
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2


N = 2
lim = 3.5 #AU
jump = 1 #number of frames skipped when plotting to speed it up
#for high tf and low steps, jump should be very close to 0
off = AU/10000 #relatively small number to avoid division by 0

ti = 0
tf = year*5
steps = 2000 #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
time = np.linspace(ti, tf, steps)
frames = steps/jump

#__________________________Code for N random bodies____________________________

#Define masses according to some initial mass distribution, usually
# taken to be around star freq \propto mass^-2

seed = random.randrange(1000)
rng = random.Random(seed)
print("Seed was:", seed) #for repeatablility


vals = np.loadtxt('v_test_vals.txt', delimiter=',')
masses = vals[0, 0:N]*Msun
x0  = vals[1, 0:N]*AU
y0  = vals[2, 0:N]*AU
vx0 = vals[3, 0:N]
vy0 = vals[4, 0:N]

init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index


#__________________Main ODE, vectorised________________________________________

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


#_________________________Energy calculations__________________________________

def T(v, masses):
    return .5 * np.sum(masses*v**2, axis=1)

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

#__________________________________Solve ODE___________________________________
#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8 #Smaller -> more accurate

sol = odeint(ode, init, time, args = (masses, G, off, size, d), rtol = rtol)
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T


Ttot = T(vx.T, masses) + T(vy.T, masses)
Utot = U(x, y, masses)

#Find relative energy distance
Etot = Ttot + Utot
dE = (Etot - Etot[0])/Etot[0]

enerax.plot(time/year, dE, 'k')
#enerax.plot(time/year, Ttot, 'r')
#enerax.plot(time/year, Utot, 'b')


#______________________________Trim Solution___________________________________

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr)-1, numElems)).astype(int)
    return Arr[idx]

#Evenly sample the full ODE for plotting
x = shorten(x,1)/AU
y = shorten(y,1)/AU


#______________________________Initialize Plots________________________________

clrs  = ['chartreuse', 'deeppink'] 
    
#Initialize trails
trails, = animax.plot([], [], '.', c = 'lightblue', markersize = '.8')

#Initialize tuple for bodies
aBodies = [] #a for animation bodies
cBodies = [] #c for canvas - ie the static main plot
xPostns = [] #For x position as a function of time


#Fill Bodies tuple with N lines, one for each body
for j in range(N):
    #Define handles for each body with corresponding size and colour
    abody_j  = animax.plot([], [],  'o', c = clrs[j],
                           markersize = 4)[0]
    canvas_j = canvas.plot([], [],  '.', c = clrs[j], 
                           markersize = .6, lw=.6)[0]
    x_j      = xposax.plot([], [], '--', c = clrs[j], 
                                     lw = 1)[0]
    #Append these handles to their respective tuple
    aBodies.append(abody_j)
    cBodies.append(canvas_j)
    xPostns.append(x_j)



#_________________________Define plot parameters_______________________________

ylim1 = -1  
ylim2 =  1 
xlim1 = -1 
xlim2 =  1 

xlabels = ['${:.1f}$'.format(lim), '${:.1f}$'.format(lim/2), '$0.0$',
          '${:.1f}$'.format((lim/2)), '${:.1f}$'.format(lim)]
ylabels = ['${:.1f}$'.format(lim), '${:.1f}$'.format(lim/2), '$0.0$',
          '${:.1f}$'.format((lim/2)), '${:.1f}$'.format(lim)]

canvas.set(xlim = (xlim1, xlim2), ylim = (ylim1, ylim2), 
           xlabel = '$x$', ylabel = '$y$')

animax.set(xlim = (xlim1, xlim2), ylim = (ylim1, ylim2), 
           xlabel = '$x$', ylabel = '$y$')

xposax.set(xlim = (ti, tf), ylim = (xlim1, xlim2),
           xticks = [0, tf/5, 2*tf/5, 3*tf/5, 4*tf/5, tf],
           xticklabels = ['$0$','$1$','$2$','$3$','$4$','$5$' ],
           xlabel = '$t$', ylabel = '$x$')
#enerax.set(ylabel = 'd$E$\n  $[10^{-5}]$')
animax.set_ylabel(animax.get_ylabel(), rotation = 0)
xposax.set_ylabel(xposax.get_ylabel(), rotation = 0)
canvas.set_ylabel(canvas.get_ylabel(), rotation = 0)
enerax.set_ylabel(enerax.get_ylabel(), rotation = 0)

#_____________________________Set Plot Data____________________________________

#initialization for many bodies (ie more than 1)
def init():
    for body in aBodies:
        body.set_data([],[])
    return body


#loop that fills the cbod_nr-th point for all points in the tuple cBodies
for cbod_nr, body in enumerate(cBodies):
    body.set_data(x[:, cbod_nr], y[:, cbod_nr])

#r = np.sqrt(x**2 + y**2)
for xposnr, body in enumerate(xPostns):
    body.set_data(np.linspace(ti, tf, len(x)), x[:, xposnr])
    #in the above line change x to r for r plot and change label

#Center of Mass Data

COM, = animax.plot([],[], '+', markersize='8', c = 'orangered')
animax.plot(0,0, '+', markersize='8', c = 'orangered', label='center of mass')

COMx = np.sum(masses*x[:,:], axis = 1)/sum(masses)
COMy = np.sum(masses*y[:,:], axis = 1)/sum(masses)


#______________________________Animation Plot__________________________________
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


    COM.set_data(COMx[i],COMy[i]) #plots COM point
    
    #Set the animation body data 
    for abod_nr, body in enumerate(aBodies):
        body.set_data(x[i, abod_nr], y[i, abod_nr])
    animax.legend()
    return fig2


anim = animation.FuncAnimation(fig2, animate, init_func=init,
                               frames=int(frames), interval=1)

#Save animation as gif (not possible from figure GUI) and show
#anim.save('Validation.gif', dpi=200, writer=PillowWriter(fps=40))

anim












