# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:23:32 2022

@author: user
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#from sympy import *
# import matplotlib.widgets as widgets - previous saves used a slider but slow

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
canvas = plt.axes([.2,.35,.6,.6])

#______________________Define constants________________________________________

msun = 1.98892e30 #Solar mass kg
mer = 5.972e24  #Earth mass kg
year = 31557600 #Year s
au = 1.49598e13 #1AU m

G = 1 #6.67408e-11 #m3 kg-1 s-2
m1, m2 = msun, mer #kg
v0 = 2*np.pi*au/year #m/s
vx1, vy1, vz1 = 0., 0., 0.
vx2, vy2, vz2 = 0., v0, 0.
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = au, 0, 0
t0, tmax = 0, 1.5*year
dt = year/1000
t = np.linspace(t0, tmax, int((tmax-t0)/dt))

N = 2    # Number of particles

offset = 0.001    # softening length

def accel(r, M, offset):
    #positions r = [x,y,z] for all particles ie 3xN array
    x = r[:,0:1]
    y = r[:,1:2]
    z = r[:,2:3]

 	#calculates rj - ri because for ri-ri this yields 0, for i =/= j,
    #yields rj-ri =/= 0; this also, therefore, maintains vector direction
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    
    '''
    Mdiv = M/(np.sqrt(dx**2 + dy**2 + dz**2 + offset**2))**3
    ax = dx @ Mdiv
 
    ay = dy @ Mdiv
    az = dz @ Mdiv
    #pack together the acceleration components
    a = G*np.hstack((ax,ay,az))
    '''
    inv_r3 = (dx**2 + dy**2 + dz**2 + offset**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ M
    ay = G * (dy * inv_r3) @ M
    az = G * (dz * inv_r3) @ M
    a = np.hstack((ax,ay,az))
    
    return a   
    

t = 0
plotRealTime = True # switch on for plotting as the simulation goes along

#mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
mass = np.array([msun,mer])

pos = np.array([[x1, y1, z1], [x2, y2, z2]])
#pos  = np.random.randn(N,3)   # randomly selected positions and velocities
vel = np.array([[vx1, vy1, vz1], [vx2, vy2, vz2]])
#vel  = -np.random.randn(N,3)
	
# Convert to Center-of-Mass frame
#vel -= np.mean(mass * vel,0) / np.mean(mass)

# calculate initial gravitational accelerations
acc = accel(pos, mass, offset)



# Simulation Main Loop
for t in range(int(tmax/dt)):
    canvas.cla()
    # (1/2) kick
    vel += acc * dt/2.0
		
    # drift
    pos += vel * dt
		
    # update accelerations
    acc = accel(pos, mass, offset)
		
	# (1/2) kick
    vel += acc * dt/2.0
		
	# update time
    t += dt
    #canvas.plot(pos[:,0]/au*100,pos[:,1]/au*100, 'bo')
    canvas.set(xlim=(-2, 2), ylim=(-2, 2))
    canvas.plot(pos[0,0]/au,pos[0,1]/au, 'bo')	
    canvas.plot(pos[1,0]/au,pos[1,1]/au, 'ro')
    
    plt.pause(0.001)
	

    plt.show()
    





