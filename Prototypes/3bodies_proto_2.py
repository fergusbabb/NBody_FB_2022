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
mer2 = mer
year = 31557600 #Year s
au = 1.49598e11 #1AU m

G = 6.67408e-11 #m3 kg-1 s-2
m1, m2, m3 = msun, mer, mer2 #kg
v0 = 2*np.pi*au/year #m/s
vx1, vy1, vz1 = 0., 0., 0.
vx2, vy2, vz2 = 0., v0, 0.
vx3, vy3, vz3 = 0., -v0, 0.
x1, y1, z1 = 0., 0., 0.
x2, y2, z2 = au, 0., 0.
x3, y3, z3 = -au, 0., 0.
t0, tmax = 0, 1.5*year
dt = tmax/500
t = 0

N = 2    # Number of particles

off = 0.01    # softening length


M = np.array([msun, mer, mer2])
pos = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
vel = np.array([[vx1, vy1, vz1], [vx2, vy2, vz2],[vx3, vy3, vz3]])
v1 = vel[0,:]
v2 = vel[1,:]
v3 = vel[2,:]


def rmag(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    rmag = np.sqrt(x**2+y**2+z**2)
    return rmag

#a = -Gm/r**2 *rhat

def accel(r1, r2, m1, m2):
    R = r2 - r1
    a1 = -G*m2*R/(rmag(R+off)**3)
    a2 = a1*m1/m2
    return a1, a2


def KE(v1,v2, m1, m2):
    return .5*m1*sum(v1**2), .5*m2*sum(v2**2)

def PE(r, m1, m2):
    return -G*m1*m2/r 
    


def acc(pos, vel, M):#, acc):
    x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:3] #the second row keeps them in
    #vector form so dx below yields a matrix not a vector
    
    #calculates rj - ri because for ri-ri this yields 0, for i =/= j,
    #yields rj-ri =/= 0; this also, therefore, maintains vector direction
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    
    #dr = np.array([[dx, dy, dz]])
    v1, v2, v3 = vel[0,:], vel[1,:], vel[2,:]
    
    inv_r3 = (dx**2 + dy**2 + dz**2 + off**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = (dx * inv_r3) @ M
    ay = (dy * inv_r3) @ M
    az = (dz * inv_r3) @ M
    a = G*np.vstack((ax,ay,az))#np.array([[ax,ay,az]])[0,:,:]
    
    #acceleration acting on particle n is a[:,n]
    return a[:,0], a[:,1], a[:,2]
print(acc(pos, vel, M))


etot_ax.set_xlim(0, tmax)

#Simulation Main Loop
while t <= tmax:
    canvas.cla()
    # (1/2) kick
    a1, a2, a3 = acc(pos, vel, M)
    v1 += a1 * dt/2
    v2 += a2 * dt/2
    v3 += a3 * dt/2
    
    # drift
    pos[0,:] += v1*dt
    pos[1,:] += v2*dt
    pos[2,:] += v3*dt
		
    # update accelerations
    a1, a2, a3 = acc(pos, vel, M)

	
	# (1/2) kick
    v1 += a1 * dt/2
    v2 += a2 * dt/2
    v3 += a3 * dt/2
    
    #KE1, KE2 = KE(v1,v2,m1,m2)
    #PEtot = PE(rmag(pos[1,:]-pos[0,:]), m1, m2)
    
	# update time
    t += dt
    #canvas.plot(pos[:,0]/au*100,pos[:,1]/au*100, 'bo')
    canvas.set(xlim=(-2,2), ylim=(-2,2))
    canvas.plot(pos[0,0]/au,pos[0,1]/au, 'bo')	
    canvas.plot(pos[1,0]/au,pos[1,1]/au, 'ro')
    canvas.plot(pos[2,0]/au,pos[2,1]/au, 'go')
    #etot_ax.plot(t,KE1+KE2, 'r.')
    #etot_ax.plot(t,PEtot, 'g.')
    #etot_ax.plot(t,KE1+KE2+PEtot, 'k.')
    plt.pause(0.001)
	

    plt.show()









