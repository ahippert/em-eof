#!/usr/bin/env python
# coding=utf-8

#######################################
#
# interp.fdist
# 
# Generate synthetics displacement-like signals
# evolving in time and space.
#
# Written by Rémi Prebet, updated and added more
# signals by Alexandre Hippert-Ferrer.
#
# @LISTIC, USMB
#
# Last update : 03/11/2018
#
#######################################


import numpy as np
from numpy import cos, sin, tanh, pi

def volcan(r, t, r0 = 0, e = 2):
    z = (1 - np.abs(r0-r)/e)*t
    return z*(z>0)

def volcan2(r, t, modes):
    f = 3.5 # Frequencies
    f1 = 3
    if modes >=2:
        z = sin(f*pi/2*t)*cos(f*3*r*pi/2)
    if modes >=3:
        z += cos(f*pi/2*t)*cos(f*3*r*pi)
    if modes >=4:
        z += sin(f*5*pi/2*t)*0.1*cos(f*10*r*pi)
    if modes >=5:
        z += sin(f*7*pi/2*t)*0.3*sin(f*15*r*pi)
    if modes >=6:
        z += sin(f*pi/2*t)*0.1*sin(f*r*pi)
    if modes >=7:
        z += cos(f*7*pi/2*t)*0.3*cos(f*11*r*pi)
    return z

def volcan3(r, t, r0=0, e=2):
    z=(1-np.abs(r0-r)/e)*t + 0.5*(1-np.abs(r0-r)/e)*t**2
    return z*(z>0)

def volcan4(r, t, r0=0, e=2):
    z = np.abs(1 - np.abs(r0-r)/e)*t + np.sin(np.pi/2*t)*np.cos(r*np.pi/2)
    return z*(z>0)

def depla(r,t):
    return sin(r)*sin(t) + sin(2.1*r)*sin(2.1*t) \
        + sin(3.1*r)*sin(3.1*t) + tanh(r)*cos(t) \
        + tanh(2*r)*cos(2.1*t) + tanh(4*r)*cos(0.1*t) \
        + tanh(2.4*r)*cos(1.1*t) + tanh(r + t) \
        + tanh(r + 2*t)

def depla2(r1, r2, r3, r4, t, modes):
    f = 1 # Frequencies
    if modes >=2:
        z = sin(f*pi/2*t)*cos(f*2*r1*pi)
    if modes >=3:
        z += sin(f*pi/2*t)*cos(f*2*r2*pi)
    if modes >=4:
        z += cos(f*5*pi/2*t)*cos(f*r3*pi)
    if modes >=5:
        z += cos(f*7*pi/2*t)*sin(f*r4*pi/2)
    return z

def genere_topo(nt, r, topo):
    return np.array([topo(r,(t)/(nt)) for t in range(10,nt+10)])

# generate random synthetic 2D field
def deterministic_field(i, j, X, Y):
    r = (i*2*pi)/X
    t = (j*2*pi)/Y
    return sin(r)*sin(t) + sin(2.1*r)*sin(2.1*t) \
        + sin(3.1*r)*sin(3.1*t) + tanh(r)*cos(t) \
        + tanh(2*r)*cos(2.1*t) + tanh(4*r)*cos(0.1*t) \
        + tanh(2.4*r)*cos(1.1*t) + tanh(r + t) \
        + tanh(r + 2*t)

def linear_field(obs, time):
    field = np.zeros((obs,time))
    for i in range(0, obs):
        r = np.linspace(0, 10, obs)[i]
        for j in range(0, time):
            t = np.linspace(1, 100, time)[j]
            field[i][j] = (1 - 2*r)*t
    return field

def construct_field(m, n):
    field = np.zeros((m,n))
    for i in range(0, m):
        x = (i*2*np.pi)/m
        for j in range(0, n):
            t = (j*2*np.pi)/n
            field[i][j] = deterministic_field(x, t)
    return field


### TESTS ###


# # Animation
# import numpy as np
# import matplotlib.pyplot as plt

# xx = np.linspace(-1,1,v.nx)

# Tt = range(0,v.nt)
# for t in Tt:
#     plt.ion()
#     t = t/(v.nt-1)
#     plt.figure(1)
#     plt.clf()
#     plt.plot(xx,volcan3(np.abs(xx),t))
#     plt.axis([-1,1,-2,2])
#     plt.title("t={:2.2f} s".format(t))
#     plt.pause(10**-5)

# ## Affichage à t donné
# import numpy as np
# import matplotlib.pyplot as plt

# xx = np.linspace(-1,1,1000)
# t = 0
# plt.clf()
# plt.plot(xx,volcan3(np.abs(xx),t))
# plt.axis([-1,1,-1.6,1.6])
# plt.tight_layout()
# plt.savefig("sin0.png")

# aff(np.array([depla(v.r,t) for t in np.linspace(0,1,30)]), color=True)
