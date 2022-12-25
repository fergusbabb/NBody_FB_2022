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
import time

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
start_time = time.time()

fig  = plt.figure(figsize=(8, 6))
fig2 = plt.figure(figsize=(8,7))
fig3 = plt.figure(figsize=(10,8.5))

widths  = [.1, 2, .1, 2, .1, 2]
heights = [2, .1, 2, .1]

gs = fig.add_gridspec(ncols = 6, nrows = 4, width_ratios = widths,
                          height_ratios = heights)


enerax = fig.add_axes([.2, .15, .7, .75])

Axes = []
A = 6
for j in range(0, A):
    #Define handles for each body with corresponding size and colour
    if j < A/2:
        Aj = fig3.add_subplot(gs[0, 2*j+1], aspect='equal')
    elif j >= A/2:
        Aj = fig3.add_subplot(gs[2, 2*(j-3)+1], aspect='equal')
    Aj.set_facecolor('navy')
    Aj.set(xlim=(-3.5,3.5), ylim=(-5,3.5), xticks=[-2,0,2],yticks=[-4,-2,0,2])
    Axes.append(Aj)
fig3.tight_layout()
fig3.subplots_adjust(hspace = 0.3, wspace = 0.1, top=0.9, bottom = 0.05)

animax = fig2.add_axes([.15, .15, .7, .7], aspect='equal')
animax.set_facecolor('navy')



#_________________________Define Constants and conditions______________________
print('Defining constants and initial conditions...')

AU = 1.496e11 #m
M = 1.989e30 #kg
day = 60*60*24 #s
year = day*365.25 #s
G = 6.67408e-11 #m3 kg-1 s-2

tau = np.sqrt((AU**3)/(G*M)) #one unit of time
N = 3
lim = 3.5 #AU
jump = 100 #number of frames skipped when plotting to speed it up
#for high tf and low steps, jump should be very close to 0
off = AU/1e11 #relatively small number to avoid division by 0

ti = 0
tf = tau*60 #To match the reference article
F = 200000 #Large number of steps
steps = int(F*N*tf/year) #too small -> singularity, too large -> too slow,
#needs to be proportional to total time and number of particles
times = np.linspace(ti, tf, steps)
st = int(steps/(jump*6))
frames = 2**8

vals = np.loadtxt('Burrau_vals.txt', delimiter=',')
masses = vals[0, 0:N]*M
x0  = vals[1, 0:N]*AU
y0  = vals[2, 0:N]*AU 
vx0 = vals[3, 0:N]
vy0 = vals[4, 0:N]
init = np.append(np.append(np.append(x0, y0), vx0), vy0)
size = int(len(init))
d = int(size/4) #defines smallest needed index


#__________________Main ODE, vectorised________________________________________
print('Solving ODE...')

def ode(init, times, masses, G, off, size, d):
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

sol = odeint(ode, init, times, args = (masses,G,off,size,d), rtol=rtol/5000)
x = sol[:,0:d]
y = sol[:,d:2*d]
vx = sol[:,2*d:3*d].T
vy = sol[:,3*d:4*d].T

#_________________________Energy calculations__________________________________
print('Finding energies...')

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

enerax.plot(times/tau, dE*1e8, 'k')

#______________________________Trim Solution___________________________________

def shorten(Arr, jump):
    numElems = int(len(Arr)/jump)
    idx = np.round(np.linspace(0, len(Arr)-1, numElems)).astype(int)
    return Arr[idx]

#Evenly sample the full ODE for plotting
x = shorten(x,jump)/AU
y = shorten(y,jump)/AU

#______________________________Initialize Plots________________________________
print('Initializing plots...')

clrs  = ['chartreuse', 'deeppink', 'darkorange' ] 
    
#Initialize trails
trails, = animax.plot([], [], '.', c = 'lightblue', markersize = '.8')

#Initialize tuple for bodies
aBodies = [] #a for animation bodies
xPostns = [] #For x position as a function of time


#Fill Bodies tuple with N lines, one for each body
for j in range(N):
    #Define handles for each body with corresponding size and colour
    abody_j  = animax.plot([], [],  'o', c = clrs[j],
                           markersize = 4)[0]
    #Append these handles to their respective tuple
    aBodies.append(abody_j)




#_________________________Define plot parameters_______________________________

animax.set(xlim = (-5, 5), ylim = (-5, 5), xlabel = '$x$', ylabel = '$y$')

enerax.set(xlim = (ti, tf/tau), xticks=[0, 10, 20, 30, 40, 50, 60],
           xlabel = '$t$', ylim = (-3, 8), yticks =[-2,0, 2, 4,6,8],
           ylabel = 'd$E \ [10^{-8}]$')
animax.set_ylabel(animax.get_ylabel(), rotation = 0)
enerax.set_ylabel(enerax.get_ylabel(), rotation = 0)
enerax.yaxis.set_label_coords(-.12, .52)

#_____________________________Set Plot Data____________________________________
print('Plotting data...')

#initialization for many bodies (ie more than 1)
def init():
    for body in aBodies:
        body.set_data([],[])
    return body

alph = ['i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.'] 
#Plot for correct time interval on different plots       
for j in range(0, A):
    for i in range(0,N):
        Axes[j].plot(x[st*j:st*(j+1), i], y[st*j:st*(j+1), i],'.', c=clrs[i], 
                               markersize = .2, lw=.1)
    Axes[j].set(title='{} \ ${:.0f} \ge t \ge {:.0f}$'.format(alph[j], 
                                                         10*j, 10*(j+1)),
                xlabel='$x$', ylabel='$y$')
    Axes[j].set_ylabel(Axes[j].get_ylabel(), rotation = 0)

#Center of Mass Data
COM, = animax.plot([],[], '+', markersize = '8', c = 'orangered', label =
                   'Center of Mass')
aleg = animax.legend(facecolor = 'midnightblue', prop={'size': 18})
for legend_text in aleg.get_texts():
        legend_text.set_color('white')


#______________________________Animation Plot__________________________________
print('Animating...')
quality = 200
#To avoid plotting every single timestep (could be 1 million frames +)
fpy = quality/year #desired frames per year
frames = fpy*(tf-ti)
jump2 = int(len(x)/frames)
x = shorten(x,jump2)
y = shorten(y,jump2)
COMx = np.sum(masses*x[:,:], axis = 1)/sum(masses)
COMy = np.sum(masses*y[:,:], axis = 1)/sum(masses)

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

    return fig2


anim = animation.FuncAnimation(fig2, animate, init_func=init,
                               frames = int(frames), interval=1)

#Save animation as gif (not possible from figure GUI) and show
print('Saving Animation')
#anim.save('Burraus_test.gif', dpi=100, writer=PillowWriter(fps=40))

print('Finished.')
end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))

anim












