#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, cycler

rc('xtick', labelsize=14)
rc('ytick', labelsize=14)

def plot_time_series(X_truth, X_noise, X_reco, X_gaps, px, py):
    """Plot time series of reconstructed field at coordinates (px,py)
    Input:
      - X_truth: true field
      - X_noise: true field + noise
      - X_reco: reconstructed time series
      - X_gaps: true field + noise + gaps
      - px, py: coordinates of values to plot
    """
    nt = X_truth.shape[0]
    allEOFs = np.zeros((nt, len(X_reco)))
    plt.figure()
    a = X_truth[:,px,py]
    b = X_gaps[:,px,py]
    c = X_noise[:,px,py]
    for i in range(len(X_reco)):
        allEOFs[:,i] = X_reco[i][:,px,py]
    plt.plot(a)
    plt.plot(b, 'k')
    plt.plot(c, 'k--')
    for i in range(len(X_reco)):
        plt.plot(allEOFs[:,i])

def plot_field(X_truth, X_reco, X_gaps, noise, ind):
    """Plot field at ind position in the time series
    """
    img = [ind]
    cmaps = ['RdGy', 'inferno', 'viridis', 'seismic']
    ax2title = ['Residuals', 'Noise']
    cm = 1
    row1, col1 = 1, 5
    for k in range(len(X_reco)):
        max1, min1 = np.nanmax(X_truth[ind])+0.01, np.nanmin(X_truth[ind])-0.01
        max2, min2 = np.max(noise[ind])+0.005, np.min(noise[ind])-0.005
        off = abs(max1-min1)/2 + abs(max1)
        min1 += off
        max1 += off
        mat1 = [X_truth+off,
                X_gaps+off,
                X_reco[k]+off,
                X_gaps-X_reco[k],
                noise]
        for i in img :
            j=1
            fig, axs = plt.subplots(nrows=row1, ncols=col1, figsize=(17, 6))
            #fig.suptitle('%d EOF - %d%% gaps - SNR=%0.02f' %(k+1, percent, sn_ratio[0]), fontsize = 16)

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
