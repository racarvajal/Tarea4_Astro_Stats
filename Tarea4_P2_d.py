#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simulate multiple draws from transit + polynomial model
# Obtain distribution from Chi_square values

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from numpy import random
from matplotlib import rc

# Plot style
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('bmh')

# Import data from file
datos_file = 'datos.dat'
datos = np.loadtxt(datos_file)

sigma = 3e-5  # stdev
p = 5  # degree of polynomial
n_draws = 1e3  # number of draws from model

# Solve for complete model, then extract needed parameters
T = np.transpose(datos[:, 0])  # Time vector
Y = np.transpose(datos[:, 1])  # Y_k vector
L = np.transpose(np.ones(np.shape(datos)[0]))  # 1s vector
V_inv = sigma**-2 * np.identity(np.shape(datos)[0])  # inverse variance-covariance
M = np.zeros((np.shape(datos)[0], p + 2))  # design matrix

# Filling design matrix
M[:, 0] = [-1 if (0.4 < t < 0.7) else 0 for t in T]  # transit coefficients: delta
M[:, 1:] = np.reshape([[np.power(T[k], i) for i in np.arange(0, p + 1)] for k in np.arange(0, np.shape(datos)[0])], (np.shape(datos)[0], p + 1))  # Polynomial

# Obtain parameters from MLE
# Theta = (Y_k - M.Theta - L)^t.V^{-1}.(Y_k - M.Theta - L)
M_t = np.transpose(M)
A = inv(np.dot(np.dot(M_t, V_inv), M))
Y_L = Y - L
B = np.dot(np.dot(M_t, V_inv), Y_L)

Theta = np.dot(A, B)  # best parameters from MLE

Y_theta = np.dot(M, Theta) + L  # Y points using parameters in model

'''
# Obtain X^2 and p-value from model
# X^{2} = (Y_k - Y_theta)^t.V^{-1}.(Y_k - Y_theta)
C = (Y - Y_theta)
C_t = np.transpose(C)
X_sq = np.dot(np.dot(C_t, V_inv), C)

# P-value: P(chi_sq >= X^{2}) = 1 - P(chi_sq < X^{2})
cdf_X_sq = chi2.cdf(np.shape(datos)[0] - (p + 2), X_sq)
p_value = 1 - cdf_X_sq
'''

# Normal distribution N(0, sigma^2) to randomize polynomial model draws
eps = random.normal(loc=0.0, scale=sigma, size=(np.shape(datos)[0], n_draws))

# Generate new Y_k vector
new_Y = [np.dot(M, Theta) + L + eps[:, i] for i in np.arange(0, n_draws)]

# Obtain X^2 and p-value from model for every draw
# X^{2} = (Y_k - Y_theta)^t.V^{-1}.(Y_k - Y_theta)
C = [new_Y[i][:]  - np.dot(M, Theta) - L for i in np.arange(0, int(n_draws))]
C_t = [np.transpose(C[i][:]) for i in np.arange(0, int(n_draws))]
X_sq_new = [np.dot(np.dot(C_t[:][i], V_inv), C[:][i]) for i in np.arange(0, int(n_draws))]

# P-values: P(chi_sq >= X^{2}) = 1 - P(chi_sq < X^{2})
cdf_X_sq = [chi2.cdf(np.shape(datos)[0] - (p + 2), X_sq_new[:][i]) for i in np.arange(0, int(n_draws))]
p_values = np.ones(n_draws) - cdf_X_sq

# Plot histogram for p-values
num_bins = 10
plt.hist(p_values, num_bins, histtype='bar', color='grey', label=('$n=%i$' % n_draws))
plt.xlabel('$\mathrm{p-value}$')
plt.ylabel('$\mathrm{Frequency}$')
plt.legend(loc='best', title='$\mathrm{transit}+\mathrm{polynomial}$')
# plt.savefig('T4_p2_d.pdf')
plt.show()

# Plot one of the draws
'''
plt.errorbar(datos[:, 0], new_Y[:][1], yerr=sigma, ls='none', marker='p')
plt.errorbar(datos[:, 0], datos[:, 1], yerr=datos[:, 2], ls='none', marker='p', label='$\mathrm{Original}$')
plt.plot(datos[:, 0], Y_theta)
plt.xlim(0.15, 0.85)
plt.ylim(0.9997, 1.0002)
ax = plt.gca()
ax.set_yticklabels(ax.get_yticks())
plt.xlabel('$\mathrm{tiempo}$ $\mathrm{(dias)}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.show()
'''
