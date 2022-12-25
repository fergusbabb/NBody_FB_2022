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

fig  = plt.figure(figsize=(16.5,10))
fig2 = plt.figure(figsize=(12,10))

widths  = [.1, .1, 2, 2]
heights = [2,  .2, 1]
gs = fig.add_gridspec(ncols = 4, nrows = 3, width_ratios = widths,
                          height_ratios = heights)

colbar = fig.add_subplot(gs[0, 0])
canvas = fig.add_subplot(gs[0, 2], aspect='equal')
surfax = fig.add_subplot(gs[0, 3])
xposax = fig.add_subplot(gs[2, :])

animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')

canvas.set_facecolor('navy')
animax.set_facecolor('navy')
xposax.set_facecolor('navy')

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
N1 = int(np.floor(3*N/5))
N2 = N - N1
lim = 15 #AU
quality = 1000 #higher is better, but too high leads to very slow animation
off = lim*AU/100 #relatively small number to avoid division by 0

ti = 0
tf = year
steps = int(750*N*tf/year) #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
time = np.linspace(ti, tf, steps)
BH1mass = Msun*1e3
BH2mass = Msun*5e2

#__________________________Code for N random bodies____________________________

#Define masses according to some initial mass distribution, usually
# taken to be around star freq \propto mass^-2

seed = random.randrange(1000)
rng = random.Random(seed)
print("Seed was:", seed) #for repeatablility


def f(x):
    return (x**-1.2)

X = np.linspace(Mmin, Mmax, N1)
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

m1 = dist(N1).reshape(N1, 1) #Variable masses
m1[0] = BH1mass
m2 = dist(N2).reshape(N2, 1) #Variable masses
m2[0] = BH2mass
masses = np.append(m1, m2)
masses_noBH = np.append(m1[1:N1], m2[1:N2])

#Define cluster parameters
sigx1, sigy1 = AU*3, AU*3
locx1, locy1 = 8*AU, 5*AU

sigx2, sigy2 = AU*3, AU*3
locx2, locy2 = -8*AU, -5*AU

#Gaussians for each cluster
xr1 = np.random.normal(size=N1, loc=0, scale=sigx1)
yr1 = np.random.normal(size=N1, loc=0, scale=sigy1)
xr2 = np.random.normal(size=N2, loc=0, scale=sigx2)
yr2 = np.random.normal(size=N2, loc=0, scale=sigy2)

for i in range(0, N1):
    xr1[i]*= (-1)**i
    yr1[i]*= (-1)**i
for j in range(0, N2):
    xr2[j]*= (-1)**j
    yr2[j]*= (-1)**j
    
xr1 += locx1
yr1 += locy1
xr2 += locx2
yr2 += locy2

#Append clusters to 1 array each
x0 = np.append(xr1, xr2)
y0 = np.append(yr1, yr2)

#Define location of BHs to be at center of clusters
x0[0] = locx1
y0[0] = locy1
x0[N1] = locx2
y0[N1] = locy2
dist = np.sqrt((locx2-locx1)**2 + (locy2-locy1)**2)

def orbvels(m1, m2, period, distance):
    M = m1+m2
    T = period
    a = ((G*M*T**2)/(4*np.pi**2))**1/3
    r = distance
    v = np.sqrt(G*M*(2/r - 1/a))
    return v

v = orbvels(np.sum(m1), np.sum(m2), year/4, dist)
v_av1 = v*np.sum(m1)/np.sum(masses)
v_av2 = v - v_av1

def v0(x, y, ap_mass, N):
    #GMm/r = mv^2/2
    U0x, U0y = np.zeros(N), np.zeros(N)
    for i in range(1,N):
        U0x[i] = G*ap_mass/x[i]
        U0y[i] = G*ap_mass/y[i]
    vx0 = np.sign(x)*np.sqrt(abs(2*U0x))
    vy0 = np.sign(y)*np.sqrt(abs(2*U0y))
    return vx0, vy0

vx1, vy1 = v0(xr1, yr1, np.mean(masses[0:N1]), N1)
vx2, vy2 = v0(xr2, yr2, np.mean(masses[N1:N]), N-N1)
vx0 = np.append(vx1, vx2)
vy0 = np.append(vy1, vy2)

angle1, angle2 = 9*np.pi/8, np.pi/8

vx0[0],   vy0[0]   =  v_av1*np.cos(angle1),  v_av1*np.sin(angle1)
vx0[N1],  vy0[N1]  =  v_av2*np.cos(angle2),  v_av2*np.sin(angle2)
vx0[1:N1]+=vx0[0]
vy0[1:N1]+=vy0[0]
vx0[N1+1:N]+=vx0[N1]
vy0[N1+1:N]+=vy0[N1] #To make the bodies orbit their BH

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


#_________________________Energy calculations__________________________________

'''
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
'''


#__________________________________Solve ODE___________________________________
#relative tolerance because want control over accuracy everywhere
rtol = 1.49012e-8 #Smaller -> more accurate

sol = odeint(ode, init, time, args = (masses, G, off, size, d), rtol = rtol/10)
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T

'''
hX, hY = x.copy(), y.copy()
hX0 = hX[:,0]/AU
hxg, tg = np.meshgrid(time, hX0)

surfax.pcolormesh(time/year, hxg, time*hxg)'''

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
sizes.insert(N1, 5)
clrs.insert(0,  'white')
clrs.insert(N1, 'white')
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
    x_j      = xposax.plot([], [], '--', c = clrs[j], 
                                     lw = sizes[j]/4)[0]
    
    #Append these handles to their respective tuple
    aBodies.append(abody_j)
    cBodies.append(canvas_j)
    xPostns.append(x_j)


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

xposax.set(xlim = (ti, tf), ylim = (-lim, lim),
           xticks = [tf/4, tf/2, 3*tf/4, tf],
           xticklabels = ['${:.2f}$'.format(tf/(4*year)),
                          '${:.2f}$'.format(tf/(2*year)),
                          '${:.2f}$'.format(3*tf/(4*year)),
                          '${:.2f}$'.format(tf/year)],
           yticks = ticks, yticklabels = labels, 
           xlabel = '$t$ [yrs]', ylabel = '$x$ [AU]')

'''surfax.set(xlim = (ti, tf), ylim = (-lim, lim),
           xticks = [tf/4, tf/2, 3*tf/4, tf],
           xticklabels = ['${:.2f}$'.format(tf/(4*year)),
                          '${:.2f}$'.format(tf/(2*year)),
                          '${:.2f}$'.format(3*tf/(4*year)),
                          '${:.2f}$'.format(tf/year)],
           yticks = ticks, yticklabels = labels, 
           xlabel = '$t$ [yrs]', ylabel = '$x$ [AU]')
'''
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

#r = np.sqrt(x**2 + y**2)
for xposnr, body in enumerate(xPostns):
    body.set_data(np.linspace(ti, tf, len(x)), x[:, xposnr])
    #in the above line change x to r for r plot and change label

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
#anim.save('{}Bodies_2BHclusters.gif'.format(N), dpi=200, 
#          writer=PillowWriter(fps=40))

#anim












