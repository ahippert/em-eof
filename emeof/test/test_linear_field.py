#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import math
from scipy import signal
import time
from scipy.stats import gaussian_kde, pearsonr, norm

from emeof import fields, miss, noises, stats, graphics
from emeof import eof_reconstruction as eof_r

##################### PARAMETERS #####################

nx, ny = 200, 200  # Field space dimensions
nobs = nx*ny       # Nb of points in each field
nt = 30            # Field time dimension

t_corr = 1.1       # Correlation coeff for temporal noise. Exponent in time-corr function f(r) = 1/(r)**t_corr
s_corr = 0.5       # Correlation coeff for spatial noise in ]0,1[
pct_gaps = 30      # Percentage of gaps in data
nb_cv_points = 30  # Number of cross validations points per image
field_type = 'linear'
number_of_modes = 1

# Grid space to define images
gridx, gridy = np.linspace(-1,1,nx), np.linspace(-1,1,ny)
x, y = np.meshgrid(gridx, gridy)

# Grillage des distances a l'origine
h1, k1 = 0.1, 0.1
grid1 = np.sqrt(x**2+y**2)
grid2 = np.sqrt((x-h1)**2+(y-k1)**2)
grid3 = np.exp(-(x+y)**2) + x*y + np.tan(x)

# Init some lists that will contain errors, fields, eigenvalues
rms_all = np.array([])
sn_ratio, neofs, rms_eof, rms_cv, X_reco, eigvals, rmseof, = [],[],[],[],[],[],[]
col = True # False: temporal mean / True: spatial mean

##################### GENERATE FIELD #####################

# Generate time series of displacement/geophysical fields of size (nt,nx,ny)
X_truth = fields.generate_field_time_series(nt, grid1, field_type, number_of_modes)

# Generate noise time series of size (nt,nx,ny)
noise = noises.generate_noise_time_series(nt, nx, ny, t_corr, s_corr, 'white')

# Total displacement field
expo = 0.5 # Tune this to augment or lower noise
data = X_truth + noise*expo
datai = np.reshape(data, (nt, nobs)).T # form initial spatio temporal field

# Compute SNR ratio
sn_ratio.append(np.std(X_truth)/np.std(noise))

# Make some copies of the data
datai_cp = copy.copy(datai)
fdispl = copy.copy(datai)

# Generate mask where values are missing
mask0 = miss.generate_gaps_time_series(nt, nx, ny, pct_gaps, 'correlated')

# Generate mask for cross validation
mask_cv, mask_total = miss.gen_cv_mask(mask0, nb_cv_points)

# SIMULATE UNWRAPPING ERROR
# s = norm.ppf(80/100., np.mean(gaps), np.std(gaps))
# unwrap_mask = miss.gen_correlated_gaps(gaps2, s, 0, 1)
# unwrap_mask = np.reshape(unwrap_mask.astype(float), (nt, nobs)).T
# fdispl[:,25][unwrap_mask[:,0]==True] = np.reshape(truth_unwrap[25]+noises[n][25], (nx*ny))[unwrap_mask[:,0]==True]

##################### RECONSTRUCTION #####################

# Step 1: find optimal number of EOFs
init_value = stats.compute_mean(fdispl, col)
nopt, X_reco = eof_r.find_first_estimate(fdispl, datai_cp,
                                         mask_total, mask_cv,
                                         init_value, col)
# Step 2: refine missing data
beta = [0.1]
isbeta = False
fdispl_cp = copy.copy(fdispl)
X_reco, err = eof_r.reconstruct_field(fdispl_cp, datai_cp,
                                      mask_total, mask_cv,
                                      nopt, init_value, beta, isbeta, col)
num_of_estimated_eof = len(X_reco) # Optimal number of EOFs
num_of_expected_eof = 1
assert num_of_estimated_eof == num_of_expected_eof

# Add mean to the anomaly
for i in range(len(X_reco)):
    X_reco[i] = np.reshape(X_reco[i], (nx, ny, nt)).T

##################### GRAPHICS #####################

# Reshape data matrix into a time series of images for plotting
fdispl[mask_total == True] = np.nan
fdispl = np.reshape(fdispl, (nx, ny, nt)).T
fdispl_cp = np.reshape(fdispl_cp, (nx, ny, nt)).T
datai_cp = np.reshape(datai_cp, (nx, ny, nt)).T

# Plot time series
px, py = 20, 20 # coordinates of value to plot
graphics.plot_time_series(X_truth, datai_cp, X_reco, fdispl, px, py)

# Plot fields
img_number = 10 # i-th image to plot
graphics.plot_field(X_truth, X_reco, fdispl, noise, img_number, num_of_estimated_eof, pct_gaps, sn_ratio[0], field_type)

plt.show()

