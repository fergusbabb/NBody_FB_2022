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
from matplotlib import cm

#These are just the presets I like to use
plt.rc('axes', labelsize=16, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

#plt.rcdefaults()



#______________________________Initialize figures______________________________

fig  = plt.figure(figsize=(16,9))
fig2 = plt.figure(figsize=(8,7))

widths  = [2, 2]
heights = [2,  .2, 1]
gs = fig.add_gridspec(ncols = 2, nrows = 3, width_ratios = widths,
                          height_ratios = heights)

canvas = fig.add_subplot(gs[0, 0], aspect='equal')
surfax = fig.add_subplot(gs[0, 1], aspect='equal')
xposax = fig.add_subplot(gs[2, :])

animax = fig2.add_axes([.15, .15, .7, .7])

canvas.set_facecolor('navy')
animax.set_facecolor('navy')
surfax.set_facecolor('navy')
xposax.set_facecolor('navy')


#_________________________________Define Constants_____________________________

AU = 1.496e11 #m
Msun = 1.989e30 #kg
Rsun = 6.9634e8
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
#c = 299792458 #m/s
Mmin = Msun/10
Mmax = Msun*100 
#masses can actually be larger due to skewing of randomiser with IMD

N = 11
lim = 2 #AU
quality = 1000 #higher is better, but too high leads to very slow animation
off = lim*AU/100 #relatively small number to avoid division by 0

ti = 0
tf = year
steps = int(750*N*tf/year) #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
time = np.linspace(ti, tf, steps)

#__________________________Code for Sun and Earth______________________________

masses  = np.loadtxt('solarsys_masses.txt', delimiter=',')[0:N]
x0      = np.loadtxt('solarsys_x0.txt',     delimiter=',')[0:N]
y0      = np.loadtxt('solarsys_y0.txt',     delimiter=',')[0:N]
vx0     = np.loadtxt('solarsys_vx0.txt',    delimiter=',')[0:N]
vy0     = np.loadtxt('solarsys_vy0.txt',    delimiter=',')[0:N]

sol_i = [0, 3] #sun, earth
N = len(sol_i)
x0, y0   = x0[sol_i],  y0[sol_i]
vx0, vy0 = vx0[sol_i], vy0[sol_i]
masses = masses[sol_i].reshape(N,1)
#masses[-1] = Msun/2

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
    return .5*sum(masses*v**2)

def U(x, y, masses):
    x = x.reshape(steps, d, 1)
    y = y.reshape(steps, d, 1)

    #Gives all relative distances
    dx = np.swapaxes(x, 1, 2) - x
    dy = np.swapaxes(y, 1, 2) - y
    dr = np.sqrt(dx**2 + dy**2 + off**2)
    dr_in = 1/dr
    mM = np.triu(-(masses*masses.T)*dr_in, 1)
    U = G*np.sum(np.sum(mM, axis = 1), axis = 1)
    return U


#__________________________________Solve ODE___________________________________
#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8 #Smaller -> more accurate

sol = odeint(ode, init, time, args = (masses, G, off, size, d), rtol = rtol/10)
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T


#______________________________Trim Solution___________________________________


#To avoid plotting every single timestep (could be 1 million frames +)
fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump = int(len(x)/frames)

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr)-1, numElems)).astype(int)
    return Arr[idx]

#Evenly sample the full ODE for plotting
x = shorten(x,jump)/AU
y = shorten(y,jump)/AU

#_____________________________Potential Surface________________________________

pts = 2**10 - 25

sxmin = .995*AU
sxmax = 1.005*AU
symin = -.005*AU
symax = .005*AU

xt = np.linspace(sxmin, sxmax, pts)
yt = np.linspace(symin, symax, pts)

xt, yt = np.meshgrid(xt, yt)


def Usurf_func(x, y):
    return -G/(np.sqrt((xt-x)**2 + (yt-y)**2 + off))

X = (x[0,:].reshape(N))*AU
Y = (y[0,:].reshape(N))*AU

Usurf = np.zeros(shape=(pts,pts))

for i in range(0, N):
    Usurf += masses[i]*Usurf_func(X[i], Y[i]) #x, y are the mass positions
    
    

zmin = -G*Msun/(.98*AU) #Potential at r = Rsun

for i in range(len(xt)):
    for j in range(len(yt)):
        if Usurf[i, j] <= zmin:
            Usurf[i, j] = zmin



surfax.contour(xt/AU, yt/AU, -np.log2(-Usurf), levels = 22, cmap = cm.cool)
surfax.plot(1, 0, 'o', c = 'aqua', markersize='10')

surfax.set(xlim = (sxmin/AU, sxmax/AU), ylim = (symin/AU, symax/AU),
           xlabel = '$x$ [AU]', ylabel = '$y$ [AU]')

#______________________________Initialize Plots________________________________


sizes = [5, 2]
clrs  = ['gold', 'aqua'] 
    

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
                           markersize = sizes[j]*5/4)[0]
    canvas_j = canvas.plot([], [],  '.', c = clrs[j], 
                           markersize = sizes[j]*1/4)[0]
    x_j      = xposax.plot([], [], '--', c = clrs[j], 
                                     lw = sizes[j]/2)[0]
    
    #Append these handles to their respective tuple
    aBodies.append(abody_j)
    cBodies.append(canvas_j)
    xPostns.append(x_j)


#_________________________Define plot parameters_______________________________

ticks = [-lim, -lim/2, 0, lim/2, lim]
labels = ['$-{:.1f}$'.format(lim), '$-{:.1f}$'.format(lim/2), '$0.0$',
          '${:.1f}$'.format((lim/2)), '${:.1f}$'.format(lim)]

canvas.set(xlim = (-lim, lim), ylim = (-lim, lim), xticks = ticks, 
           xticklabels = labels, yticks = ticks, yticklabels = labels, 
           xlabel = '$x$ [AU]', ylabel = '$y$ [AU]')

animax.set(xlim = (-lim, lim), ylim = (-lim, lim), xticks = ticks, 
           xticklabels = labels, yticks = ticks, yticklabels = labels, 
           xlabel = '$x$ [AU]', ylabel = '$y$ [AU]')

xposax.set(xlim = (ti, tf), ylim = (-lim, lim),
           xticks = [tf/4, tf/2, 3*tf/4, tf],
           xticklabels = ['${:.2f}$'.format(tf/(4*year)),
                          '${:.2f}$'.format(tf/(2*year)),
                          '${:.2f}$'.format(3*tf/(4*year)),
                          '${:.2f}$'.format(tf/year)],
           yticks = ticks, yticklabels = labels, 
           xlabel = '$t$ [yrs]', ylabel = '$x$ [AU]')

animax.set_yticklabels(animax.get_yticklabels(), rotation = 45)
animax.set_xticklabels(animax.get_xticklabels(), rotation = 45)
canvas.set_yticklabels(canvas.get_yticklabels(), rotation = 45)
canvas.set_xticklabels(canvas.get_xticklabels(), rotation = 45)
   


#_____________________________Set Plot Data____________________________________

#initialization for many bodies (ie more than 1)
def init():
    for body in aBodies:
        body.set_data([],[])
    return body


#loop that fills the cbod_nr-th point for all points in the tuple cBodies
for cbod_nr, body in enumerate(cBodies):
    body.set_data(x[:, cbod_nr], y[:, cbod_nr])

for xposnr, body in enumerate(xPostns):
    body.set_data(np.linspace(ti, tf, len(x)), x[:, xposnr])



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
    
    #Set the animation body data 
    for abod_nr, body in enumerate(aBodies):
        body.set_data(x[i, abod_nr], y[i, abod_nr])

    return fig2


anim = animation.FuncAnimation(fig2, animate, init_func=init,
                               frames=int(frames), interval=1)

#Save animation as gif (not possible from figure GUI) and show
#anim.save('Sun-Earth.gif', dpi=300, 
#          writer=PillowWriter(fps=20))

anim












