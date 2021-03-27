#!/usr/bin/env python

import numpy as np
from numpy import cos, sin, tanh, pi

# generate random synthetic 2D field
def deterministic_field(i, j, X, Y):
    r = (i*2*pi)/X
    t = (j*2*pi)/Y
    return sin(r)*sin(t) + sin(2.1*r)*sin(2.1*t) \
        + sin(3.1*r)*sin(3.1*t) + tanh(r)*cos(t) \
        + tanh(2*r)*cos(2.1*t) + tanh(4*r)*cos(0.1*t) \
        + tanh(2.4*r)*cos(1.1*t) + tanh(r + t) \
        + tanh(r + 2*t)

# generate linear displacement field
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
