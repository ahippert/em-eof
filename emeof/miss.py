#!/usr/bin/env python

# gaps_gen.py
# Generate random or space-correlated gaps (missing values) in a given dataset
#
# AH 05/09/2018
# Last update 23/11/2018

import numpy as np

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

def gen_cv_mask(mask, N, gaps, verbose=False):
    """Generate mask for cross validation points (same principle than gen_random_gaps())
    """
    for k in gaps :
        val = np.random.randint(0, N)
        try :
            while mask[val]:
                if verbose : print ('redundancy found')
                val = np.random.randint(0, N)
        except ValueError :
            if verbose : print ("no redundancy found")
        mask[val] = True
    return mask

def mask_field(displ, mask, fval):
    """Apply mask on array and fill masked values with user-defined value
    """
    return np.ma.filled(np.ma.array(displ, mask=mask, fill_value=0.0), fill_value=fval)

""" Gen holes directly on spatio_temporal matrix """
# X = np.reshape(field_s.T, (nb_obs, nb_time))
# mask_0 = np.zeros((nb_obs, nb_time), dtype=bool)
# mask = holes.gen_holes_mask(mask_0, nb_obs, nb_time, tirage, remise, holes_list, verbose=False)
# field_zeros = np.ma.filled(np.ma.array(X, mask=mask, fill_value=0.0),
#                            fill_value=fill_val)
# field_A = copy.copy(field_zeros)
