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
plt.rc('axes', labelsize=20, titlesize=18)
plt.rc('figure')#, autolayout=True)  # Adjusts supblot parameters for new size

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#plt.rcdefaults()



#______________________________Initialize figures______________________________

fig  = plt.figure(figsize=(12,9))
fig2 = plt.figure(figsize=(12,10))

widths  = [.1, .4, 4]
heights = [1, 4,  1]
gs = fig.add_gridspec(ncols = 3, nrows = 3, width_ratios = widths,
                          height_ratios = heights)

colbar = fig.add_subplot(gs[1, 0])
canvas = fig.add_subplot(gs[:, 2], aspect='equal')



animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')

canvas.set_facecolor('navy')
animax.set_facecolor('navy')

#_________________________________Define Constants_____________________________

AU = 1.496e11 #m
Msun = 1.989e30 #kg
#Rsun = 6.9634e8
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2
#c = 299792458 #m/s
Mmin = Msun/10
Mmax = Msun*100
#masses can actually be larger due to skewing of randomiser with IMD

N = 10
lim = 4 #AU
quality = 2000 #higher is better, but too high leads to very slow animation
off = lim*AU/100 #relatively small number to avoid division by 0

ti = 0
tf = year/4
steps = int(750*N*tf/year) #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
time = np.linspace(ti, tf, steps)
BH1mass = Msun*1e3


#__________________________Code for N random bodies____________________________

#Define masses according to some initial mass distribution, usually
# taken to be around star freq \propto mass^-2

seed = random.randrange(1000)
rng = random.Random(seed)
print("Seed was:", seed) #for repeatablility


def f(x):
    return (x**-1.2)

X = np.linspace(Mmin, Mmax, N)
Y = f(X)

def dist(N):
    m = np.random.uniform(Mmin, Mmax, N)
    freq = np.random.uniform(0, np.max(Y), N) #Note any coefficient in the 
    #original Y makes no difference - the only relevant thing is the trend
    j = 0
    while j < N:
        if freq[j] <= f(m[j]):
            j += 1
        elif freq[j] > f(m[j]):
            freq[j] = np.random.uniform(0,np.max(Y))
            m[j] = np.random.uniform(Mmin, Mmax)
    return m

m1 = dist(N).reshape(N, 1) #Variable masses
m1[0] = BH1mass
masses = m1
masses_noBH = m1[1:N]

#Define cluster parameters
sigx1, sigy1 = AU, AU
locx1, locy1 = 0, 0


#Gaussians for each cluster
xr1 = np.random.normal(size=N, loc=0, scale=sigx1)
yr1 = np.random.normal(size=N, loc=0, scale=sigy1)


for i in range(0, N):
    xr1[i]*= (-1)**i
    yr1[i]*= (-1)**i

xr1 += locx1
yr1 += locy1


#Append clusters to 1 array each
x0 = xr1
y0 = yr1

#Define location of BHs to be at center of clusters
x0[0] = locx1
y0[0] = locy1



def v0(x, y, ap_mass, N):
    #GMm/r = mv^2/2
    U0x, U0y = np.zeros(N), np.zeros(N)
    for i in range(1,N):
        U0x[i] = G*ap_mass/x[i]
        U0y[i] = G*ap_mass/y[i]
    vx0 = np.sign(x)*np.sqrt(abs(2*U0x)/2)
    vy0 = np.sign(y)*np.sqrt(abs(2*U0y)/2)
    return vx0, vy0

vx0, vy0 = v0(xr1, yr1, np.mean(masses[0:N]), N)



vx0[0], vy0[0]  =  0, 0
vx0[1:N]+=vx0[0]
vy0[1:N]+=vy0[0] #To make the bodies orbit their BH

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
    
    a = G*np.append(ax, ay)*0.999 #equivalent to the original vx, vy
    return np.append(v, a) #returns 1D array of v, a


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


#______________________________Initialize Plots________________________________

#Define the segment of the colourmap I am using
#mapmin, mapmax define the range
mapmin1, mapmax1 = .15, .3
mapmin2, mapmax2 = .4, .92

mapmin, mapmax = mapmin1, mapmax2
interval1 = np.linspace(mapmin1, mapmax1, 80)
interval2 = np.linspace(mapmin2, mapmax2, 150)
interval = np.append(interval1, interval2)

colors = plt.cm.gist_ncar(interval) #Use colourmap 'gist_ncar' w/in 'interval'
colormap = LinearSegmentedColormap.from_list('name', colors)

#l defines the normalized mass range - NOT masses in Solar Masses
l = (masses/np.max(masses))
#s defines the sizing scale so larger mass gives larger marker etc
s = 1 + abs(2*np.log10(Msun/masses) + np.min(np.log10(Msun/masses)))
l = mapmin + (mapmax - mapmin)*l  
#Gets the mass data into a relative array from 1 to 0 to assign colour to mass


sizes = []
clrs  = [] 

for k in range(len(l)):
    #Define colour and markersize dependent on the mass of the body
    clrs.append(colormap(l[k]))
    sizes.append(s[k])
    
#Append black hole values
sizes.insert(0, 7.5)
clrs.insert(0,  'white')
#Initialize trails
trails, = animax.plot([], [], '.', c = 'lightblue', markersize = '.4')

#Initialize tuple for bodies
aBodies = [] #a for animation bodies
cBodies = [] #c for canvas - ie the static main plot
xPostns = [] #For x position as a function of time

#Fill Bodies tuple with N lines, one for each body
for j in range(N):
    #Define handles for each body with corresponding size and colour
    abody_j  = animax.plot([], [],  'o', c = clrs[j],
                           markersize = sizes[j]/2)[0]
    canvas_j = canvas.plot([], [],  '.', c = clrs[j], 
                           markersize = sizes[j]/4)[0]
    
    #Append these handles to their respective tuple
    aBodies.append(abody_j)
    cBodies.append(canvas_j)

for i in range(0,N):
    canvas.plot(x[0,i], y[0,i], 'X', c = clrs[i], markersize = sizes[i])
    
    
#generate a hidden plot to sample the colours from - ie generate colourbar
sm = plt.cm.ScalarMappable(cmap=colormap,
        norm=LogNorm(vmin = np.min(masses/Msun),
                           vmax = np.max(masses/Msun)))
sm.set_array([])  


#_________________________Define plot parameters_______________________________

ticks =  [-lim, -lim/2, 0, lim/2, lim]
ticks2 = [0, lim/2, lim, 3*lim/2, 2*lim]
labels = ['$-{:.1f}$'.format(lim), '$-{:.1f}$'.format(lim/2), '$0.0$',
          '${:.1f}$'.format((lim/2)), '${:.1f}$'.format(lim)]
labels2 = ['$0.0$', '${:.1f}$'.format(lim/2), '${:.1f}$'.format(lim),
          '${:.1f}$'.format((3*lim/2)), '${:.1f}$'.format(2*lim)]

canvas.set(xlim = (-lim, lim), ylim = (-lim, lim), xticks = ticks, 
           xticklabels = labels, yticks = ticks, yticklabels = labels, 
           xlabel = '$x$ [AU]', ylabel = '$y$ [AU]')

animax.set(xlim = (-lim, lim), ylim = (-lim, lim), xticks = ticks, 
           xticklabels = labels, yticks = ticks, yticklabels = labels, 
           xlabel = '$x$ [AU]', ylabel = '$y$ [AU]')

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


#Plot colourbar
cbar = plt.colorbar(sm, cax = colbar, label = '$M_{\odot}$')
cbar.ax.set_title('Mass Key', y=1.05)



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
anim.save('{}Bodies_singlecluster.gif'.format(N), dpi=100, 
          writer=PillowWriter(fps=40))

#anim












