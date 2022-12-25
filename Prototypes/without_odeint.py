# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:19:02 2022

@author: user
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
#from sympy import *

# These are just the presets I like to use
mpl.style.use('classic')
plt.rc('axes', labelsize=20, titlesize=20)
plt.rc('figure', autolayout=True)  # Adjusts supblot parameters for new size
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})


#_________________________Define figures and axes______________________________
#Main figure

fig = plt.figure(figsize=(20,10))
fig.set_tight_layout(False)

#Plots
canvas = plt.axes([.2,.35,.6,.6],aspect='equal')
etot_ax = plt.axes([.1,.075,.8,.2])
#______________________Define constants________________________________________

msun = 1.98892e30 #Solar mass kg
mer = 5.972e24  #Earth mass kg
year = 31557600 #Year s
au = 1.49598e11 #1AU m

G = 6.67408e-11 #m3 kg-1 s-2
m1, m2 = msun, mer #kg
v0 = 2*np.pi*au/year #m/s
vx1, vy1, vz1 = 0., 0., 0.
vx2, vy2, vz2 = 0., v0, 0.
x1, y1, z1 = 0., 0., 0.
x2, y2, z2 = au, 0., 0.
x2, y2, z2 = -au, 0., 0.
t0, tmax = 0, 1.5*year
dt = tmax/500
t = 0

N = 2    # Number of particles

offset = 0.001    # softening length


mass = np.array([msun,mer])
pos = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
vel = np.array([[vx1, vy1, vz1], [vx2, vy2, vz2],[vx3, vy3, vz3]])
v1 = vel[0,:]
v2 = vel[1,:]



def rmag(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    rmag = np.sqrt(x**2+y**2+z**2)
    return rmag

#a = -Gm/r**2 *rhat

def accel(pos1, pos2, m1, m2):
    R = pos2 - pos1
    a1 = -G*m2*R/(rmag(R)**3)
    a2 = a1*m1/m2
    return a1, a2


def KE(v1,v2, m1, m2):
    return .5*m1*sum(v1**2), .5*m2*sum(v2**2)

def PE(r, m1, m2):
    return -G*m1*m2/r 
    

etot_ax.set_xlim(0, tmax)

# Simulation Main Loop
while t <= tmax:
    canvas.cla()
    # (1/2) kick
    a1, a2 = accel(pos[0,:],pos[1,:], m1, m2)
    v1 += a1 * dt/2
    v2 += a2 *dt/2
    
    # drift
    pos[0,:] += v1*dt
    pos[1,:] += v2*dt
		
    # update accelerations
    a1, a2 = accel(pos[0,:],pos[1,:], m1, m2)
		
	# (1/2) kick
    v1 += a1 * dt/2
    v2 += a2 *dt/2
    
    KE1, KE2 = KE(v1,v2,m1,m2)
    PEtot = PE(rmag(pos[1,:]-pos[0,:]), m1, m2)
    
	# update time
    t += dt
    #canvas.plot(pos[:,0]/au*100,pos[:,1]/au*100, 'bo')
    canvas.set(xlim=(-4, 4), ylim=(-4, 4))
    canvas.plot(pos[0,0]/au,pos[0,1]/au, 'bo')	
    canvas.plot(pos[1,0]/au,pos[1,1]/au, 'ro')
    etot_ax.plot(t,KE1+KE2, 'r.')
    etot_ax.plot(t,PEtot, 'g.')
    etot_ax.plot(t,KE1+KE2+PEtot, 'k.')
    plt.pause(0.001)
	

    plt.show()









