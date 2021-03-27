#!/usr/bin/env python

# Script main.py
#
# This code performs EOF interpolation of a synthetic spatio-temporal field
# containing missing data.
#
# Author : Alexandre Hippert-Ferrer , LISTIC
# Created : 04/2018
# Last update : AH 29/04/2019

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, cycler
import copy
import math
from scipy import signal
import time
from scipy.stats import gaussian_kde, pearsonr, norm

from emeof import fields, miss, noises, stats
from emeof import eof_reconstruction as eof_r

rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

#__________________________
#__________________________

# Field dimensions
nx, ny = 200, 200
nobs = nx*ny # i from 1... m : nb of points in each map

# Time dimension
nt = 30
x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny)) # Grillage de l'espace
h1, k1 = 0.1, 0.1
r1 = np.sqrt(x**2+y**2)
r2 = np.sqrt((x-h1)**2+(y-k1)**2) # Grillage des distances a l'origine
r5 = np.exp(-(x+y)**2) + x*y + np.tan(x)

r = r1

# Create multiple frequency image
#x1, y1 = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny//2)) #Grillage de l'espace
tinter = np.linspace(0,500,nt)
freq2 = np.array([fields.volcan2(r2, (t/nt)+30/nt, 5) for t in range(10,nt+10)])#tinter
freq2 += np.array([fields.volcan(r2, (t/nt)+30/nt) for t in tinter])#range(10,nt+10)])
freq3 = np.array([fields.volcan2(r5, (t/nt)+30/nt, 3) for t in range(10, nt+10)])
freq4 = np.array([fields.volcan2(r1, (t/nt)+30/nt, 5) for t in range(10, nt+10)])
# Time series of displacement-like field
mode = 1
modes = 1 #in [2,7]
if mode == 1:
    truth = np.array([fields.volcan(r, (t/nt)+30/nt) for t in range(10, nt+10)])
elif mode == 2:
    truth = np.array([fields.volcan(r, (t/nt)+30/nt) for t in range(10, nt+10)])#tinter
    truth += np.array([fields.volcan2(r, (t/nt)+30/nt, modes) for t in range(10,nt+10)])#tinter
elif mode > 2:
    truth = np.array([fields.deterministic_field(r, (t/nt)+30/nt) for t in range(10, nt+10)])
else:
    truth = np.array([
        [fields.deterministic_field(i, j, nobs, nt)
         for j in range(nt)]
        for i in range(nobs)])
truth_unwrap = np.array([fields.volcan(r2, (t/nt)+30/nt) for t in range(10, nt+10)])

# seismic displacement in LOS
#truth = np.load("seismic5000.npy")

# Construct and tune noise
rms_all = np.array([])
sn_ratio = []
neofs, neofs1 = [], []
rms = []
rms_eof = []
rms_cv = []
fields = []
eigvals = []
rmseof = []

# re-grid
x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny)) #Grillage de l'espace
rad = np.sqrt(x**2+y**2)

# Construct other noise useful for correlated gaps generation
mu, sigma = 0, 1
b = np.random.normal(mu, sigma, (nx, ny))
b2 = np.random.normal(mu, sigma, (nx, ny))
e = np.linspace(1.3, 1.4, nt)
e2 = np.linspace(1.5, 1.9, nt)
corr = [noises.geo(rad, e[i]) for i in range(nt)]
corr2 = [noises.geo(rad, e2[i]) for i in range(nt)]
gaps = noises.gen_noise_series2(corr, b, nt)
gaps2 = noises.gen_noise_series2(corr2, b2, nt)

col = True # False: temporal mean / True: spatial mean

# Generate noise
BUILDNOISE = True
NOISE_TYPE = 'stcorr' # 'rand' or 'scorr' or 'stcorr'
if not BUILDNOISE:
    noise = np.zeros((nt,nx,ny))
else:
    blanc = np.random.normal(mu, sigma, (nt, nx, ny))
    expo = 1.1 # exponent in correlation function (as: 1/(r)**expo)
    r = np.arange(0, 1, 0.005)
    geo_noise = noises.geo(rad, expo)
    if NOISE_TYPE == 'stcorr':
        noise = noises.gen_noise_series(geo_noise, blanc, nt) + noises.gen_corr_noise(0.5,
                                                                              (nt,nx,ny),
                                                                              0)
    elif NOISE_TYPE == 'scorr':
        noise = noises.gen_noise_series(geo_noise, blanc, nt)
    else:
        noise = blanc
# time series of evolving noise
inter = 10
mul = [0.5]
noises = [noise*i for i in mul]
pct = inter

#run = np.arange(1,20,1)
exp = 1
for z in range(exp):
    # n = 0
    data = truth + noises[0] # total displacement field
    datai = np.reshape(data, (nt, nobs)).T # form initial spatio temporal field
    if z==0:
        sn_ratio.append(np.std(truth)/np.std(noises[n]))

    datai_cp = copy.copy(datai)

    dataimean = stats.compute_mean(datai, col)

    gens = ['correlated']#, 'correlated'] #'random' 'correlated'
    pourcent = np.linspace(3, 80, pct)

    k = 50
    for gen in gens:
        fdispl = copy.copy(datai)
        # 1. generate correlated gaps using noise
        if gen == 'correlated':
            t_start, t_end = 8, 18
            seuil = norm.ppf(k/100., np.mean(gaps), np.std(gaps))
            print ('seuil: %0.2f' %seuil)
            mask0 = miss.gen_correlated_gaps(gaps, seuil, t_start, t_end)
            mask0 = np.reshape(mask0, (nt, nobs)).T

        # 2. Generate random gaps
        elif gen == 'random':
            ngaps = np.arange(int(nobs*nt*k/100.))
            mask0 = miss.gen_random_gaps(np.zeros((nobs, nt), dtype=bool),
                                         nobs,
                                         nt,
                                         ngaps)
            for i in range(nt):
                print(len(mask0[:,i][mask0[:,i]==True]))
        # mask for cross validation
        ngaps = [30]
        ngaps_cv = [np.arange(i) for i in ngaps]
        #for m in range(len(ngaps)):
        tng = ngaps[0]*nt
        mask_temp = copy.copy(mask0)
        for i in range(nt):
            mask_temp[:,i] = miss.gen_cv_mask(mask_temp[:,i],
                                              nobs,
                                              ngaps_cv[0])

        # Generate mask for cross validation
        mask_cv = np.logical_xor(mask_temp, mask0)

        # Create mask where data exists for later use
        mask_data = np.invert(mask_temp)
        n_data = len(mask_data[mask_data==True])

        # Apply mask on displacement field
        #fdispl = gg.mask_field(fdispl, mask_temp, np.nan)


        # SIMULATE UNWRAPPING ERROR
        s = norm.ppf(80/100., np.mean(gaps), np.std(gaps))
        unwrap_mask = miss.gen_correlated_gaps(gaps2, s, 0, 1)
        unwrap_mask = np.reshape(unwrap_mask.astype(float), (nt, nobs)).T
        fdispl[:,25][unwrap_mask[:,0]==True] = np.reshape(truth_unwrap[25]+noises[n][25], (nx*ny))[unwrap_mask[:,0]==True]

        fcp = copy.copy(fdispl)

        # Reconstruction procedure
        # Step 1 : find optimal number of EOFs
        init_value = stats.compute_mean(fdispl, col)
        cv_tab = [ngaps[0] for j in range(nt)]
        nopt, fields = eof_r.find_first_estimate(fdispl, datai_cp,
                                                 mask_temp, mask_cv,
                                                 init_value, col)
        # Step 2 : refine missing data
        beta = [0.1]
        isbeta = False
        fdispl_cp = copy.copy(fdispl)
        fields, err = eof_r.reconstruct_field(fdispl_cp, datai_cp,
                                              mask_temp, mask_cv,
                                              nopt, init_value, beta, isbeta, col)
        neof1 = len(fields) # Optimal number of EOFs
        neofs1.append(neof1)



# Add mean to the anomaly
for i in range(len(fields)):
    fields[i] = np.reshape(fields[i], (nx, ny, nt)).T

# # Add temporal mean that was substracted
# #ftruth = stats.add_mean(ftruth, truth_mean, col)
# fdispl = stats.add_mean(fdispl, dataimean, col)
# datai_cp = stats.add_mean(datai_cp, dataimean, col)

# # reshape data matrix into a time series of images
fdispl[mask_temp == True] = np.nan
fdispl = np.reshape(fdispl, (nx, ny, nt)).T
fdispl_cp = np.reshape(fdispl_cp, (nx, ny, nt)).T
datai_cp = np.reshape(datai_cp, (nx, ny, nt)).T
fcp = np.reshape(fcp, (nx, ny, nt)).T


"""""""""""""""""""""---GRAPHICS---"""""""""""""""""""""

# Energy bar plots

#eigvals.append(np.linalg.eigvals(np.cov(field.T)))
#variance = compute_variance_by_mode(nt, eigval)
#plot_bars(range(1, nt+1), variance, 'Mode', 'Contained variance in mode k (%)')

##### ERROR PLOTS #####
# if len(nts) > 1 :
#     plt.figure(2)
#     plt.xticks(range(0, len(nts)), nts)
#     plt.plot(range(0, len(nts)), rms_nt, 'k-o', linewidth=1)
#     plt.title('Root Mean Square Error vs. number of images')

# print (rms_vc)
# fig, ax = plt.subplots()
# #plt.xticks(ngaps)
# ax.plot(100*(ngaps/nobs), rms_cv, 'k-')
# ax.plot(100*(ngaps/nobs), rms_eof, 'r-')
# ax.set_xlabel('% of point used in cross validation per image')
# ax.set_ylabel('RMSE')
# ax.set_title('Interpolation error vs. number of points used in ross validation')

## RMSE versus Iterations ##
# plt.figure()
# plt.title('RMSE vs iterations')
# plt.plot(rmscv[:-1], label='cross-v error')
# plt.plot(rmseof[:-1], label='real error')
# plt.legend()

# cmap = plt.cm.coolwarm
# rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(beta))))
# fig, ax = plt.subplots()
# for i in range(len(beta)):
#     ax.plot(run, neofs[i::len(beta)], label=str(np.float32(beta[i])))
#     ax.legend()
# plt.show()

# COMPARE WITH OTHER METHODS
# fig, ax = plt.subplots()
# ax.plot(sn_ratio, rmsem_mean/exp, 'b--', label='EM-EOF') #pourcent
# ax.plot(sn_ratio, rmsnn_mean/exp, 'k-', label='NNI') #pourcent
# ax.plot(sn_ratio, rmskr_mean/exp, 'k--', label='SK') #pourcent
# ax.set_ylabel('RMSE')
# ax.set_xlabel('SNR')
# # #ax.legend()
# plt.show()

## EIGENVALUES PLOTS ##
# plt.figure()
# for i in range(len(eigvals)):
#     plt.plot(range(nt), np.sort(np.real(eigvals[i]))[::-1])

## TIME SERIES PLOTS ##
px, py = 20, 20
indice = 25
a, b, c = [], [], []
tplt = 30
allEOFs = np.zeros((tplt, len(fields)))
plt.figure()
a = truth[:tplt,px,py]
b = fdispl[:tplt,px,py]
c = datai_cp[:tplt,px,py]
for i in range(len(fields)):
    allEOFs[:,i] = fields[i][:tplt,px,py]
plt.plot(a)
plt.plot(b, 'k')
plt.plot(c, 'k--')
for i in range(len(fields)):
    plt.plot(allEOFs[:,i])


##### plot fields #####
ind = 10
img = [ind]
cmaps = ['RdGy', 'inferno', 'viridis', 'seismic']
ax2title = ['Residuals', 'Noise']
cm = 1
row1, col1 = 1, 5
for k in range(len(fields)):
    max1, min1 = np.nanmax(truth[ind])+0.01, np.nanmin(truth[ind])-0.01
    max2, min2 = np.max(noises[0][ind])+0.005, np.min(noises[0][ind])-0.005
    off = abs(max1-min1)/2 + abs(max1)
    min1 += off
    max1 += off
    mat1 = [truth+off,
            fdispl+off,
            fields[k]+off,
            fdispl-fields[k],
            noises[0]]
    for i in img :
        j=1
        fig, axs = plt.subplots(nrows=row1, ncols=col1, figsize=(17, 6))
        fig.suptitle('%d EOF - %d%% gaps - SNR=%0.02f' %(k+1, pourcent[0], sn_ratio[n]), fontsize = 16)

        im1 = axs.flat[0].imshow(mat1[0][i], vmin=min1, vmax=max1, cmap = cmaps[cm])
        axs.flat[0].axis('off')
        fig.subplots_adjust(right=0.8, wspace=0.1)

        for ax in axs.flat[1:3]:
            im = ax.imshow(mat1[j][i].T, vmin=min1, vmax=max1, cmap = cmaps[cm])
            ax.axis('off')
            ax.get_yaxis().set_visible(False)
            j+=1
            fig.subplots_adjust(right=0.8, wspace=0.1)
        cb_ax = fig.add_axes([0.15, 0.15, 0.35, 0.05])
        cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')

        im1 = axs.flat[3].imshow(mat1[3][i].T, vmin=min2, vmax=max2, cmap = cmaps[cm])
        axs.flat[3].axis('off')
        axs.flat[3].get_yaxis().set_visible(False)
        cb_ax1 = fig.add_axes([0.56, 0.15, 0.2, 0.05])
        cbar = fig.colorbar(im1, cax=cb_ax1, orientation='horizontal')

        im2 = axs.flat[4].imshow(mat1[4][i], vmin=min2, vmax=max2, cmap = cmaps[cm])
        axs.flat[4].axis('off')
        axs.flat[4].get_yaxis().set_visible(False)

plt.show()
