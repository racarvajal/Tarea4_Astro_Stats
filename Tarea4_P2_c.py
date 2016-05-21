#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Simulate multiple draws from different degree polynomial model
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
# p = 5  # original degree of polynomial
par_num = 3  # number of polynomial degrees to test
degrees = [2, 5, 15]  # polynomial degrees to test. By hand
n_draws = 1e3  # number of draws from model

# Values for polynomial model
T = np.transpose(datos[:, 0])  # Time vector
Y = np.transpose(datos[:, 1])  # Y_k vector
L = np.transpose(np.ones(np.shape(datos)[0]))  # 1s vector
V_inv = sigma**-2 * np.identity(np.shape(datos)[0])  # inverse of variance-covariance
p_values = np.zeros((n_draws, par_num))  # Matrix for all p-values distributions

# Loop in three different values of p
for (j,p) in zip(np.arange(0, par_num), degrees):
	# Values for polynomial model
	M = np.zeros((np.shape(datos)[0], p + 1))  # Design matrix dim = (p + 1, p + 1)

	# Fill design matrix M_ik = (y_k)^i
	M[:, :] = np.reshape([[np.power(T[k], i) for i in np.arange(0, p + 1)] for k in np.arange(0, np.shape(datos)[0])], (np.shape(datos)[0], p + 1))

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
	cdf_X_sq = chi2.cdf(np.shape(datos)[0] - (p + 1), X_sq)
	p_value = 1 - cdf_X_sq
	'''
	# Generate draws using obtained parameters
	eps = random.normal(loc=0.0, scale=sigma, size=(np.shape(datos)[0], n_draws))  # errors
	new_Y = [np.dot(M, Theta) + L + eps[:, i] for i in np.arange(0, n_draws)]  # from model with errors

	# Obtain X^2 and p-value from model for all draws
	# X^{2} = (Y_k - Y_theta)^t.V^{-1}.(Y_k - Y_theta)
	C = [new_Y[i][:]  - np.dot(M, Theta) - L for i in np.arange(0, int(n_draws))]
	C_t = [np.transpose(C[i][:]) for i in np.arange(0, int(n_draws))]
	X_sq_new = [np.dot(np.dot(C_t[:][i], V_inv), C[:][i]) for i in np.arange(0, int(n_draws))]

	# P-value: P(chi_sq >= X^{2}) = 1 - P(chi_sq < X^{2})
	cdf_X_sq = [chi2.cdf(np.shape(datos)[0] - (p + 1), X_sq_new[:][i]) for i in np.arange(0, int(n_draws))]
	p_value = np.ones(n_draws) - cdf_X_sq
	p_values[:, j] = np.transpose(p_value)

# Plot histogram for p-values
num_bins = 10
# n, bins, patches = plt.hist(p_value, num_bins, histtype='step', color='grey')
plt.hist(p_values[:, 2], num_bins, histtype='bar', color='grey', label=('$p=%i$' % degrees[2]), alpha=0.95, hatch='o')
plt.hist(p_values[:, 1], num_bins, histtype='bar', color='blue', label=('$p=%i$' % degrees[1]), alpha=0.55, hatch='x')
plt.hist(p_values[:, 0], num_bins, histtype='bar', color='red', label=('$p=%i$' % degrees[0]), alpha=0.55)
plt.xlabel('$\mathrm{p-value}$')
plt.ylabel('$\mathrm{Frequency}$')
plt.legend(loc='best', title=('$n=%i$' % n_draws))
# plt.savefig('T4_p2_c.pdf')
plt.show()

'''
# Plot one of the draws
plt.errorbar(datos[:, 0], new_Y[:][1], yerr=sigma, ls='none', marker='p', label='$\mathrm{Modified}$')
plt.errorbar(datos[:, 0], datos[:, 1], yerr=datos[:, 2], ls='none', marker='p', label='$\mathrm{Original}$')
plt.plot(datos[:, 0], Y_theta)
plt.xlim(0.15, 0.85)
plt.ylim(0.9997, 1.0002)
ax = plt.gca()
ax.set_yticklabels(ax.get_yticks())
plt.xlabel('$\mathrm{tiempo}$ $\mathrm{(dias)}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc='best')
plt.show()
'''
