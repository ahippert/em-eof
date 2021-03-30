#!/usr/bin/env python

###############################################
# noise.py
#
# Generates correlated noise in space and/or time
# using Cholesky decomposition of a defined covariance
# matrix.
#
# @Alexandre Hippert-Ferrer & Remi Prebet
# Created : 08/10/2018
# Last updated : 03/11/2018
###############################################
import numpy as np

# function taken from R. Prebet
def recentre(im, A, B):
    """Applique une transformation affine pour que les valeurs de im soient entre A et B (A < B)"""
    M, m = np.max(im), np.min(im)
    return (im-m)/(M-m)*(B-A) + A

def aff_transform(val, a, b, c, d):
    return (val-a)/(b-a)*(d-c) + c

def geo(r, b):
    #global gam
    corr = 1/(r)**(b)
    gam = np.abs(np.fft.fft2(corr))**2
    return gam/np.max(gam)

def geo_1D(r, b):
    #global gam
    corr = 1/(r)**(b)
    gam = np.abs(np.fft.fft(corr))**2
    return gam/np.max(gam)

def exp(r, b):
    corr = np.exp(-b*r)
    gam = np.abs(np.fft.fft2(corr))**2
    return gam/np.max(gam)

def gen_noise(corr, alea):
    filt = corr*np.fft.fft2(alea)
    inv = np.real(np.fft.ifft2(filt))
    eps1, eps2 = 0.04*np.random.random(2)-0.02
    inv = recentre(inv,-0.4+eps1,0.4+eps2) # amp du bruit ici
    return (inv - np.mean(inv))

def gen_noise_1D(corr, alea):
    filt = corr*np.fft.fft(alea)
    inv = np.real(np.fft.ifft(filt))
    eps1, eps2 = 0.04*np.random.random(2)-0.02
    inv = recentre(inv,-0.4+eps1,0.4+eps2) # amp du bruit ici
    return (inv - np.mean(inv))

def gen_noise2(corr, alea, coeff=0.05):
    filt = corr*np.fft.fft2(alea)
    inv = np.real(np.fft.ifft2(filt))
    eps1, eps2 = coeff*np.random.random(), coeff*np.random.random()
    return recentre(inv,eps1,1-eps2)

def generate_noise_time_series(nt, nx, ny, t_corr, s_corr, noise_type):
    """ Generate noise time series of multiple types
    Input:
      - corr: degree of space correlation
      - nt: time dimension
      - nx, ny: size of each field
      - noise_type: 'scorr' (spatially correlated),
                    'stcorr'(spatio-temporally correlated)
                    'white' (white Gaussian noise with mean=0,sig=1)
    Output:
      - A nt-length time series of noise fields of size nx*ny
    """

    # Set mean and sigma and generate white gaussian noise
    mu, sigma = 0, 1
    alea = np.random.normal(mu, sigma, (nt, nx, ny))

    # Noise generation
    if noise_type in ['scorr', 'stcorr']: # Depends on noise_type
        x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
        rad = np.sqrt(x**2+y**2)
        geo_noise = geo(rad, t_corr)
        noise = np.array([gen_noise(geo_noise, alea[i]) for i in range(0, nt)])
        if noise_type == 'stcorr': # If spatio-temporal, add space correlated noise
            noise += gen_corr_noise(s_corr,(nt,nx,ny),0)
    elif noise_type == 'white':
        noise = alea
    else:
        print("Not a valid type of noise")
    return noise

def gen_noise_series1(corr, alea, time):
    return np.array([gen_noise_1D(corr, alea[i]) for i in range(0, time)])

def gen_noise_series2(corr, alea, time):
    return np.array([gen_noise(corr[i], alea) for i in range(0, time)])

def temporal_noise():
    return None

def gen_corr_noise(r, shape, corr):
    """Generates time series of correlated noise using
    Cholesky decomposition
    Input :
      - shape : tuple of ints, shape of the desired noise matrix
      - r : correlation coefficient in ]0,1[
      - corr : type of correlation
           0 --> time-correlated noise
           1 --> space-correlated noise
    Output:
      - y : noise matrix of dimension (nt, ns) or (ns, nt)
        depending on the type of correlation
    """
    if corr==0:
        R = gen_covariance_matrix(shape[0], r)
    else:
        R = gen_covariance_matrix(shape[1], r)
    L = np.linalg.cholesky(R)
    x = np.random.normal(0, 0.1, (shape[0],shape[1]*shape[2]))
    y = np.dot(L, x)
    return np.reshape(y,shape) #y.T

def gen_covariance_matrix(m, corcoef):
    """Generates covariance matrix for a given
    correlation correlation coefficient
    Input :
      - m : size of covariance matrix (should be symmetric
        and positive semi definite)
      - corcoef : correlation coefficient in ]0,1[ interval
    Output :
      - cov : Covariance matrix
    """
    cov = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cov[i][j] = corcoef**(abs(i-j))
    #R = np.kron(Rt, Rs)
    return cov

def apply_patch_on_image(patches, nt, ns, pw):
    """Apply patches on a time series
    Input :
     - patches : array of patches in format (nt, ns, ns)
     - nt : time dimension (number of images)
     - ns : space dimension (number of points)
     - pw : patch width
    Output:
     - img : time series of patches
    """
    img = np.zeros((nt, ns*pw, ns*pw))
    for i in range(nt):
        for j in range(ns):
            for k in range(ns):
                for c in range(ns):
                    img[i][j+c*ns][ns*k:ns*(k+1)] = patches[k+c*ns][i][j]
    return img

def gen_monotone_pattern(r, n, p, perc):
    return 0
