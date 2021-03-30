#!/usr/bin/env python

# gaps_gen.py
# Generate random or space-correlated gaps (missing values) in a given dataset
#
# AH 05/09/2018
# Last update 23/11/2018

import numpy as np
from scipy.stats import norm
import copy
from . import noises

def gen_random_gaps(mask, x, y, gaps, verbose=False):
    """Generate mask of random missing data
    """
    for k in gaps :
        xgap = np.random.randint(0, x)
        ygap = np.random.randint(0, y)
        try :
            while (mask[xgap, ygap] == True) :
                if verbose : print("redundancy found for mask[%d, %d]" % (xgap, ygap))
                xgap = np.random.randint(0, x)
                ygap = np.random.randint(0, y)
                if verbose : print("new gap at : mask[%d, %d]" % (xgap, ygap))
        except ValueError :
            if verbose : print ("no redundancy found")
        mask[xgap, ygap] = True
    return mask

def gen_correlated_gaps(field, seuil, tstart, tend):
    """Generate mask of missing data correlated in space
    A threshold on the given field's low values is applied to give
    the mask its shape
    """
    mask = np.zeros((field.shape[0], field.shape[1], field.shape[2]), dtype=bool)
    for i in range(field.shape[0]):
        if i >= tstart and i <= tend :
            mask[i][field[i] < seuil] = True
    return mask

def gen_periodic_gaps(field, seuil, tstart, tend):
    # Same than previous function but using high values threshold
    # [not used]
    #
    mask = np.zeros((tend-tstart, field.shape[1], field.shape[2]), dtype=bool)
    for i in range(tstart, tend):
        mask[i-tstart][field[i] > seuil] = True
    return mask

def gen_cv_mask(mask, n_gaps, verbose=False):
    """Generate mask for cross validation points (same principle than gen_random_gaps())
    """
    gaps = np.arange(n_gaps)
    mask_tmp = copy.copy(mask)
    p, nt = mask.shape
    for j in range(nt):
        for k in gaps :
            val = np.random.randint(0, p)
            try :
                while mask_tmp[val,j]:
                    if verbose : print ('redundancy found')
                    val = np.random.randint(0, p)
            except ValueError :
                if verbose : print ("no redundancy found")
            mask_tmp[val,j] = True
    mask_cv = np.logical_xor(mask_tmp, mask)
    return mask_cv, mask_tmp

def mask_field(displ, mask, fval):
    """Apply mask on array and fill masked values with user-defined value
    """
    return np.ma.filled(np.ma.array(displ, mask=mask, fill_value=0.0), fill_value=fval)

def generate_gaps_time_series(nt, nx, ny, pct, gaps_type):
    """ Generate gaps on time series of modeled fields.
    Input:
      - pct: percentage of gaps to be generated
      - gaps_type: type of gaps, can be:
                    - 'random': randomly distributed gaps across time series
                    - 'corr': spatio-temporally correlated gaps (gaps with shapes)
    Output:
      - p*nt mask with NaN where values are missing (p=nx*ny)
    """
    p = nx*ny
    if gaps_type == 'correlated':
        t_start, t_end = 8, 18
        x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
        rad = np.sqrt(x**2+y**2)
        b = np.random.normal(0, 1, (nx, ny))
        e = np.linspace(1.3, 1.4, nt)
        corr = [noises.geo(rad, e[i]) for i in range(nt)]
        gaps = noises.gen_noise_series2(corr, b, nt)
        seuil = norm.ppf(pct/100., np.mean(gaps), np.std(gaps))
        print ('seuil: %0.2f' %seuil)
        mask = gen_correlated_gaps(gaps, seuil, t_start, t_end)
        mask = np.reshape(mask, (nt, p)).T

    # 2. Generate random gaps
    elif gaps_type == 'random':
        nb_of_gaps = np.arange(int(p*nt*pct/100.))
        zeros_mask = np.zeros((p, nt), dtype=bool)
        mask = gen_random_gaps(zeros_mask, p, nt, nb_of_gaps)
    return mask

