#!/usr/bin/env python

# Module eof_reconstruction.py
#
#
#
# @ahippert 21/11/2018
# last updated : 07/07/2019

import numpy as np
import time
import copy
import sys
from . import stats

def eof_decomp(field, eigvec, neof):
    """ Performs EOF decomposition of a spatio-temporal field
    step 1 : retrieve principal components
    step 2 : reconstruct field using k EOFs
    """
    pcomp = [field @ eigvec[:,j] for j in range(len(eigvec))]
    field_r = np.zeros((field.shape[0], len(eigvec)))
    for k in range(neof):
        field_r += np.dot(np.expand_dims(pcomp[k], axis=0).T,
                        np.expand_dims(eigvec[:,k].T, axis=0))
    return field_r

def fast_eof_decomp(data, u, v, k):
    """ Optimized version (100 times faster)
    Performs EOF decomposition of a spatio-temporal field
    step 1 : retrieve principal components data@u
    step 2 : reconstruct field using k EOFs
    """
    # Much faster than using np.vstack((D@E)[:,k])@v[:k]
    return (data@u).T[:k].T@v[:k]

def reconstruct_per_EOF_hand(X, mask, neof):
    fields = []
    col = True
    for i in range(1,neof+1):
        for k in range(30):
            X_tmp = copy.copy(X)

            # Compute mean and substract it to data
            mean = np.mean(X_tmp, axis=0)
            X_anom = X_tmp[:] - mean

            # Retrieve eigenvector/values from SVD
            eigv, eigval, eigvt = np.linalg.svd(np.cov(X_anom.T),
                                                full_matrices=False)

            X_anom = fast_eof_decomp(X_anom, eigv, eigvt, i)
            X_tmp = stats.add_mean(X_anom, mean, col)

            # Replace missing values by estimated values
            X[mask] = X_tmp[mask]
        fields.append(X_tmp)
    return fields


def reconstruct_field(field, datai, mask, mask_cv, nopt, init, beta, optim, cov_type):
    """ Stage 2 of the EM-EOF method.
    Iterative update of the missing values.
    """
    econv = 1e-5
    e = [np.inf, 1e4]
    i, j = 1, 0
    itr = 0
    neof = 1
    N, p = field.shape
    fields, rms_cv, rmscv = [], [], []
    fcopy = copy.copy(field) # Unchanged field except missing values
    start_t = time.time()
    while e[i] < e[i-1]:
        field_tp = copy.copy(fcopy)

        # Compute mean and substract it to data
        mean = np.mean(field_tp, axis=0)
        anomaly = field_tp[:] - mean

        # Covariance computation (SCM)
        sigma = np.cov(anomaly.T)

        # Perform SVD on SCM
        eigv, eigval, eigvt = np.linalg.svd(sigma, full_matrices=False)

        anomaly = fast_eof_decomp(anomaly, eigv, eigvt, neof)
        field_tp = anomaly[:] + mean

        # rms computation in function of neof &/or iterations
        rms_cv.append(stats.rmse_cross_v(field_tp[mask_cv],
                                         datai[mask_cv],
                                         len(mask_cv[mask_cv==True])))
        fcopy[mask] = field_tp[mask]

        print('%0.08f' %rms_cv[j])
        rmscv.append(rms_cv[j])

        # Algorithm to stop reconstruction
        if rms_cv[j] > e[i]:
            end_t = time.time()
            print("Procedure stopped ! Error augmented.")
            break
        j += 1
        itr += 1
        if j > 1:
            if abs(rms_cv[j-1]-rms_cv[j-2]) > econv:
                continue
            else:
                e.append(rms_cv[j-1])
                if optim == True:
                    if (1 - e[i+1]/e[i]) < beta: # second criteria to avoid overfitting
                        end_t = time.time()
                        del(e[len(e)-1])
                        print('Procedure stopped ! Beta reached.')
                        break
                fields.append(field_tp)
                neof += 1
                if neof >= nopt:
                    end_t = time.time()
                    neof -= 1
                    print('Procedure stopped because M reached first estimate.')
                    break
                i += 1
                rms_cv = [] # Empty rms_cv
                j = 0 # Prepare next step

    print("%d iterations - %0.06f seconds \n" %(itr, end_t - start_t))
    print("Final Cross-RMSE: %0.08f" %min(e))
    print("Final number of EOFs for reconstruction: %d \n" %(neof-1))
    return [fields, min(e)]
    #return fields


def find_first_estimate(field, datai, mask, mask_cv, init, cov_type):
    """ Stage 1 of the EM-EOF method.
    Gives a first estimate of the optimal number of EOF modes to reconstruct
    a time series with missing data by finding the minimum cross-RMSE.
    INPUT:
    - field: time series with missing data
    - datai: copy of the time series used for cross-validation
    - mask: mask where data is missing (True), otherwise it contains False
    - mask_cv: mask of cross_validation data
    - init: initialization value
    OUTPUT:
    - n_opt: estimation of the optimal number of EOF modes
    """

    rms_cv = [] # will contain all cross-RMSE
    fields = []

    N, p = field.shape

    # 1st local copy
    fcopy = copy.copy(field) # Unchanged field except missing values

    for i in range(1, p+1):
        # 2nd local copy
        field_tp = copy.copy(fcopy) # Treated field (decomposition, reconstruction)

        # Compute mean and substract it to data
        mean = np.mean(field_tp, axis=0)
        anomaly = field_tp[:] - mean

        # SVD of covariance matrix
        sigma = np.cov(anomaly.T)

        #sigma = (anomaly.conj().T @ anomaly) / N
        eigv, eigval, eigvt = np.linalg.svd(sigma,
                                            full_matrices=False)

        anomaly = fast_eof_decomp(anomaly, eigv, eigvt, i)
        field_tp = anomaly[:] + mean

        # compute cross-RMSE
        rms_cv.append(stats.rmse_cross_v(field_tp[mask_cv],
                                         datai[mask_cv],
                                         len(mask_cv[mask_cv==True])))
        print(rms_cv[i-1])

        # Replace missing values by updated missing values
        fcopy[mask] = field_tp[mask]
        #print(np.mean(fcopy[mask]))

        fields += [field_tp]

    return [np.argmin(rms_cv)+1, fields] # return minimum of cross-RMSE vector

def init_missing_val(data, mask, init):
    """ Initialize missing values to a predefined
    initialization value """
    N = data.shape[1]
    for i in range(N):
        data[:,i][mask[:,i] == True] = init[i]
    return data
