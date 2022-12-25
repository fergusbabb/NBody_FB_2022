# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 22:09:12 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import random

AU = 1.496e11 #m
Msun = 1.989e30 #kg

Mmin = Msun/10
Mmax = Msun*100 
#masses can actually be larger due to skewing of randomiser with IMD

N = 200
lim = 12 #AU
quality = 1000 #higher is better, but too high leads to very slow animation
off = lim*AU/100 #relatively small number to avoid division by 0
N1 = int(np.floor(2*N/5))
N2 = N - N1

BH1mass = Msun*5e2
BH2mass = Msun*3e2

#__________________________Code for N random bodies____________________________

#Define masses according to some initial mass distribution, usually
# taken to be around star freq \propto mass^-2

seed = random.randrange(1000)
rng = random.Random(seed)
print("Seed was:", seed) #for repeatablility



def f(x):
    return (x**-1.5) #Initial mass function
X = np.linspace(Mmin, Mmax, N)/Msun
Y = f(X)

def massdist(N):    
    mr = np.random.uniform(Mmin, Mmax, N)/Msun
    f_mr = np.random.uniform(0, np.max(f(mr)), N)
    j = 0
    while j < N:
        if f_mr[j] <= f(mr[j]):
            j += 1
        elif f_mr[j] > f(mr[j]):
            f_mr[j] = np.random.uniform(0,np.max(Y))
            mr[j] = np.random.uniform(Mmin, Mmax)

    return mr, f_mr
plt.plot(massdist(N1))
cl1_m, cl1_freq = massdist(N1)
cl2_m, cl2_freq = massdist(N2)

