#!/usr/bin/env python

#########################################
#
# interp.stats
#
# Statistics in python.
#
# @alex-hf 07-09-2018
# Last update : 20-01-2020
#########################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import itertools
from scipy.linalg import toeplitz

def rmse(pred, obs, col=True):
    """Compute the Root-Mean-Square Error (RMSE)
    between prediction and observation fields
    """
    nt = pred.shape[1]
    no = pred.shape[0]
    sum_of_dif = 0
    for i in range(0, nt):
        if col :
            sum_of_dif += np.linalg.norm(pred[:,i] - obs[:,i])**2
        else :
            sum_of_dif += np.linalg.norm(pred[i] - obs[i])**2
    return np.sqrt(sum_of_dif/(nt*no))

def norm_rmse(pred, obs, N):
    """Compute normalized RMSE by the mean standard deviation
    """
    return rmse_cross_v(pred, obs, N)/np.mean(np.std(obs, axis=0))

def rmse_cross_v(pred, obs, N):
    """Compute cross-RMSE between prediction and observation fields
    """
    if N>0:
        s = np.linalg.norm(pred - obs)
        err = s/np.sqrt(N)
    else:
        err = 0
    return err

def compute_image_rmse(im1, im2, nx, ny):
    """ Compute RMSE between two images
    """
    rmse_im = 0
    for i in range(nx):
        for j in range(ny):
            rmse_im += np.linalg.norm(im1[i][j] - im2[i][j])**2
    return np.sqrt(rmse_im/(nx*ny))

def compute_mean(data, col, nozeros=True):
    """Compute the column/line mean of a matrix without NaNs
    Not used anymore, too slow compared to np.mean(x, axis=..)
    """
    datamean = []
    if not col: data = data
    else: data = data.T
    for i in range(0, data.shape[0]):
        if not nozeros:
            datamean.append(np.mean(data[i]))
        else:
            datamean.append(np.nanmean(data[i]))
    return datamean

def compute_mean0(matrix, n, col, nozeros=True):
    """Compute the column/line mean of a matrix without zeros
    """
    matrix_mean = []
    for i in range(0, n):
        if col:
            if nozeros:
                matrix_mean.append(matrix[:,i][matrix[:,i].nonzero()].mean())
            else:
                matrix_mean.append(np.mean(matrix[:,i]))
        else:
            if nozeros:
                matrix_mean.append(matrix[i][matrix[i].nonzero()].mean())
            else:
                matrix_mean.append(np.mean(matrix[i]))
    return matrix_mean

def remove_mean(data, mean, col):
    """ Remove the column/line mean of a matrix
    """
    n_data = np.zeros((data.shape[0],
                       data.shape[1]))
    if col:
        for i in range(0, data.shape[1]):
            n_data[:,i] += data[:,i]
            n_data[:,i] -= mean[i]
    else:
        for i in range(0, data.shape[0]):
            n_data[i] += data[i]
            n_data[i] -= mean[i]
    return n_data

def remove_t_mean(data, mean):
    """Remove the temporal mean of a time series matrix
    """
    n = max(data.shape[0],data.shape[1])
    for i in range(0, n):
        if data.shape[0]==n:
            data[i] -= mean[i]
        else :
            data[:,i] -= mean[i]
    return data


def remove_s_mean(data, mean):
    """Remove the spatial mean of a time series matrix
    """
    n = min(data.shape[0],data.shape[1])
    for i in range(0, n):
        if data.shape[0]==n:
            data[i] -= mean[i]
        else :
            data[:,i] -= mean[i]
    return data

def add_mean(data, mean, col):
    """ Add the column/line mean of a matrix
    """
    n_data = np.zeros((data.shape[0],
                       data.shape[1]))
    if col:
        for i in range(0, data.shape[1]):
            #if np.isnan(mean[i]):
            #    data[:,i] += 0.
            #else:
            n_data[:,i] += data[:,i]
            n_data[:,i] += mean[i]
    else:
        for i in range(0, data.shape[0]):
            #if np.isnan(mean[i]):
            #    data[i] += 0.
            #else:
            n_data[i] += data[i]
            n_data[i] += mean[i]
    return n_data


def compute_variance_by_mode(n_EOF, singular_val, cov=True):
    """Compute variance contained in any mode (EOF).
    Based on Beckers and Rixen 2001
    """
    tot_variance = 0
    variance = []
    # singular values of field matrix X are the square roots
    # of the eigenvalues of Xt*X
    if not cov :
        singular_val *= singular_val
    for i in range(0, len(singular_val)):
        tot_variance += singular_val[i]
    return [100*((singular_val[i])/tot_variance) for i in range(0, n_EOF)]


def chi2(obs, pred, n):
    """Compute statistical Chi-2 between observations and predictions
    """
    chi = 0.
    for i in range(n):
        if pred[i]==0. or obs[i]==0. or (pred[i]==0. and obs[i]!=0.):
            print ("Bad expected number in chi-2")
        else :
            chi += (1/np.var(obs))*(obs[i] - pred[i])**2
    return chi


def ftest(data1, data2):
    """Compute the F-Test between the variances of two datasets
    """
    var1, var2 = np.var(data1), np.var(data2)
    if var1 > var2:
        return var1/var2
    else:
        return var2/var1

def compute_cov(data):
    """ Estimate the sample covariance matrix of a given data set
    """
    #return (1/len(data)-1)*(data.T @ data)
    return (1/len(data)-1)*(np.matmul(data.T,data))

def autocov(v, lx):
    """ Compute a lag autocovariance.
    """
    ch = 0
    for i in range(v.shape[0]-lx):
        for j in range(v.shape[1]):
            ch += v[i][j]*v[i+lx][j]
    return ch/((v.shape[0]-lx)*v.shape[1])

def lag_autocov(v, lag):
    """ Compute lag autocovariance of a vector
    """
    tmp = 0
    N = len(v)-lag
    for j in range(N):
        tmp += v[j]*v[j+lag]
    return tmp/N

def var(x):
    """
    """
    tmp = 0
    for i in range(len(x)):
        tmp += abs(x[i]-np.mean(x))**2
    return tmp/len(x)

def autocorr(x, k):
    """ Compute lag autocorrelation in [-1,1] of a vector x
    """
    tmp = 0
    N = len(x)-k
    for i in range(N):
        tmp += (x[i]-np.mean(x))*(x[i+k]-np.mean(x))
    return tmp/(np.var(x)*len(x))

def autocorrcoeff(x, lag):
    return lag_autocov(x, lag)/np.var(x)

def rule_of_thumb(data, augmentdata, lag):
    """ Compute North et al. 1989 rule of thumb, which is an estimation
    of the typical sample error between two neighboring eigenvalues
    """
    #dataimean = compute_mean(data, col=True)
    #data = remove_mean(data, dataimean, col=True)
    n, p = augmentdata.shape
    if n>p:
        augmentdata = augmentdata.T
    else:
        p = n
    i1 = time.time()
    u, d, v = np.linalg.svd(np.cov(augmentdata), full_matrices=True)
    #print(time.time()-i1)
    #print(d)

    i2 = time.time()
    #dof1 = estim_dof(data[4960]) # Degrees of freedom
    dof1 = 3
    #print(time.time()-i2)
    #print(dof1)

    i3 = time.time()
    dof2 = estim_spatial_dof(data, lag, fast_weight(data.shape[0]))
    #print(time.time()-i3)
    print(dof2)
    #dof2 = np.array([3.8589177, 10.814314, 24.534567, 35.194347, 34.78785, 41.715157, 23.928059, 18.997124, 22.169546, 6.7797637, 26.73039, 6.9989386])

    #print(dof1)
    #print(dof2)

    dof = dof1*np.mean(dof2)
    dl = [np.log((d[i]*(2/dof)**0.5)/(d[i]-d[i+1])) for i in range(p-1)]
    dl = (max(dl)-dl)/(max(dl)-min(dl)) # Compute [0,1] confidence index
    return [dl, d, dof]

def plot_eig(d, dl, dof, optm, error):
    """
    """
    # Plot confidence index and eigenvalues
    # if error == True:
    #     fig, (ax1, ax2) = plt.subplots(2, 1)
    #     yerr = d*np.sqrt(2/dof)
    #     ax1.errorbar(np.arange(1,len(d)+1,1), d, yerr=yerr/2, fmt='ko', markersize=2, elinewidth=1, capsize=2)
    #     ax1.set_yscale('log')
    #     #ax1.annotate('(a)', xy=(390, 285), xycoords='figure points')
    #     ax1.set_ylabel("$\lambda_k$", fontsize=16)
    #     ax2.plot(dl, color='k')
    #     ax2.set_xlabel("k", fontsize=16, family='sans-serif')
    #     ax2.set_ylabel("$\mathcal{C}_k$", fontsize=16)
    #     #ax2.annotate('(b)', xy=(390, 140), xycoords='figure points')
    #     fig, ax = plt.subplots(1)
    #     ax.set_xscale('log')
    #     ax.plot(d[:optm], dl[:optm], 'o', c='orange', alpha=0.6, ms=6)
    #     ax.plot(d[optm:len(d)-1], dl[optm:], 'o', c='grey', alpha=0.5, ms=6)
    #     ax.set_xlabel("$\lambda_k$", fontsize=16)
    #     ax.set_ylabel("$\mathcal{C}_k$", fontsize=16)
    if error == True:
        lim = 10
        fig = plt.figure(constrained_layout=True)
        fsize = 15
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, :2])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        yerr = d*np.sqrt(2/dof)
        ax1.errorbar(np.arange(1,lim+1,1), d[:lim], yerr=yerr[:lim], fmt='ko', markersize=2, elinewidth=1, capsize=2)
        ax1.set_yscale('log')
        #ax1.annotate('(a)', xy=(390, 285), xycoords='figure points')
        ax1.set_ylabel("$\lambda_k$", fontsize=fsize)

        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(dl[:lim], color='k')
        ax2.set_xlabel("k", fontsize=fsize, family='sans-serif')
        ax2.set_ylabel("$\mathcal{C}_k$", fontsize=fsize)

        ax3 = fig.add_subplot(gs[0:, 2])
        ax3.set_xscale('log')
        ax3.plot(d[:optm], dl[:optm], 'o', c='orange', alpha=0.6, ms=6)
        ax3.plot(d[optm:lim], dl[optm:lim], 'o', c='grey', alpha=0.3, ms=6)
        ax3.set_xlabel("$\lambda_k$", fontsize=fsize)
        ax3.set_ylabel("$\mathcal{C}_k$", fontsize=fsize)

        #fig.suptitle("GridSpec")
        #for i, ax in enumerate(fig.axes):
        #    ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        #    ax.tick_params(labelbottom=False, labelleft=False)



def estim_dof(x):
    """ Estimate degrees of freedom following Thiebaux and Zwiers 1984
    """
    m = 0
    for k in range(1,x.shape[0]-1):
        m += (1-k/x.shape[0])*autocorr(x, k)
    return x.shape[0]/(1+2*m)

def estim_spatial_dof(x, lag, weight):
    """Spatial dof
    """
    m = 0
    for k in range(1,lag):
        m += (1-k/lag)
    corr = fast_moran(x, weight)
    # negative correlation is not taken into account
    corr[corr<0.] = 0.
    return lag/(1+2*m*corr)

def weight_matrix(n):
    """Compute a n*n weight matrix based on the inverse distance
    between indices i and j.
    """
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                W[i][j] = 1./abs(i-j)**2
            #if abs(i-j)==1:
            #    W[i][j] = 1.
            else:
                W[i][j] = 0.
        #W[i] = W[i]/np.sum(W[i])
    return W

# def fast_weight(n):
#     """ Optimized version of the weight_matrix() function
#     """
#     b = [0 for i in range(n-2)] + [1, 0, 1] + [0 for i in range(n-3)]
#     W = [[0 for i in range(n)]]
#     W[0][1] = 1.
#     for i in range(n-2):
#         W += [b[n-2-i:len(b)-i]]
#     W += [W[0][::-1]]
#     return np.array(W)

def fast_weight(n):
    """
    """
    col = np.zeros(n)
    col[1] = 1.
    return toeplitz(col, col)


def moran(z, weight):
    """Compute Moran statistics to estimate
    spatial correlation.
    """
    n = len(z)
    I, W, num = 0, 0, 0
    for i in range(n):
        for j in range(n):
            W += weight[i][j]
            I += weight[i][j]*z[i]*z[j]
        num += z[i]**2
    return (I*n)/(W*num)

def fast_moran(z, weight):
    """Optimized version of moran()
    z must be a numpy array
    """
    I = 0
    for i, j in itertools.combinations(range(len(z)), 2):
        I += weight[i][j]*z[i]*z[j]
    return (2*I*len(z))/(np.sum(weight)*np.sum(z**2))

def local_moran(z, weight):
    """Compute local Moran statistics to estimate
    spatial correlation.
    """
    n = len(z)
    I, num = 0, 0, 0
    for i in range(n):
        for j in range(n):
            I += weight[i][j]*z[i]*z[j]
        num += z[i]**2
    return I/num
