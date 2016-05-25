#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Obtain AIC and BIC for different degrees of polynomial part of model
# Select 'optimal' degree

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from numpy import random
from matplotlib import rc

# Plot style
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.style.use('bmh')

# Import data from file
datos_file = 'datos.dat'
datos = np.loadtxt(datos_file)

sigma = 3e-5  # stdev
p_max = 18  # maximum degree of polynomial
n_draws = 1e3
cdf_X_sq = np.zeros(p_max)
X_sq = np.zeros(p_max)
# AIC = np.zeros(p_max)
# BIC = np.zeros(p_max)

T = np.transpose(datos[:, 0])
Y = np.transpose(datos[:, 1])
L = np.transpose(np.ones(np.shape(datos)[0]))
V_inv = sigma**-2 * np.identity(np.shape(datos)[0])


for p in np.arange(0, p_max):

	pars = p + 2

	M = np.zeros((np.shape(datos)[0], pars))


	M[:, 0] = [-1 if (0.4 < t < 0.7) else 0 for t in T]

	M[:, 1:] = np.reshape([[np.power(T[k], i) for i in np.arange(0, pars - 1)] for k in np.arange(0, np.shape(datos)[0])], (np.shape(datos)[0], pars - 1))
	'''
	M_t = np.transpose(M)
	A = inv(np.dot(np.dot(M_t, V_inv), M))
	Y_L = Y - L
	B = np.dot(np.dot(M_t, V_inv), Y_L)
	'''
	M_t = np.transpose(M)
	A = inv(np.dot(M_t, M)) # * sigma**2
	Y_L = Y - L
	B = np.dot(M_t, Y_L)# * sigma**-2

	Theta = np.dot(A, B)

	Y_theta = np.dot(M, Theta) + L

	C = Y - Y_theta
	C_t = np.transpose(C)

	# X_sq[p] = np.dot(np.dot(C_t, V_inv), C)  # Likelihood
	X_sq[p] = np.dot(C_t, C) * sigma**-2  # Likelihood
	# cdf_X_sq[p] = chi2.cdf(np.shape(datos)[0] - pars, X_sq[p])

	# AIC[p] = -2 * np.log(X_sq[p]) + 2 * pars + (2 * pars * (pars + 1)) / (np.shape(datos)[0] - pars - 1)
	# BIC[p] = -2 * np.log(X_sq[p]) + 2 * pars + pars * np.log(np.shape(datos)[0])  # bic estimate

# AIC = [(-2 * np.log(X_sq[p]) + 2 * (p + 2) + (2 * (p + 2) * (p + 3)) / (np.shape(datos)[0] - p - 3)) for p in np.arange(0, p_max)]
AIC = [(-2 * np.log(X_sq[p]) + 2 * (p + 2)) for p in np.arange(0, p_max)]
BIC = [(-2 * np.log(X_sq[p]) + 2 * (p + 2) + (p + 2) * np.log(np.shape(datos)[0])) for p in np.arange(0, p_max)]

plt.plot(np.arange(2, p_max + 2), AIC, label='$\mathrm{AIC}$', marker='p')
plt.plot(np.arange(2, p_max + 2), BIC, label='$\mathrm{BIC}$', marker='s')
plt.xlabel('$\mathrm{p}$ : $\mathrm{parameters}$')
plt.legend(loc='best')
# plt.savefig('T4_p3_a_orig.pdf')
plt.show()


norm_AIC = [(aic - np.nanmin(AIC)) / (np.nanmax(AIC) - np.nanmin(AIC)) for aic in AIC]
norm_BIC = [(bic - np.nanmin(BIC)) / (np.nanmax(BIC) - np.nanmin(BIC)) for bic in BIC]

plt.plot(np.arange(2, p_max + 2), norm_AIC, label='$\mathrm{AIC}$', marker='p')
plt.plot(np.arange(2, p_max + 2), norm_BIC, label='$\mathrm{BIC}$', marker='s', ls='none')
plt.xlabel('$\mathrm{p}$ : $\mathrm{parameters}$')
plt.legend(loc='best')
# plt.savefig('T4_p3_a_norm.pdf')
plt.show()


'''
p_value = 1 - cdf_X_sq
print('The p-value is: %f' % p_value)
print('H_0 is rejected with a probability of: %f' % cdf_X_sq)
print('H_0 is rejected with a probability of: %f%%' % (cdf_X_sq * 100))

Theta_pol = Theta
M_pol = M
L_pol = L

eps = random.normal(loc=0.0, scale=sigma, size=(np.shape(datos)[0], n_draws))
new_Y = [np.dot(M_pol, Theta_pol) + L + eps[:, i] for i in np.arange(0, n_draws)]

C_pol = [new_Y[i][:]  - np.dot(M_pol, Theta_pol) - L for i in np.arange(0, int(n_draws))]
C_pol_t = [np.transpose(C_pol[i][:]) for i in np.arange(0, int(n_draws))]

X_sq_new = [np.dot(np.dot(C_pol_t[:][i], V_inv), C_pol[:][i]) for i in np.arange(0, int(n_draws))]
cdf_X_sq_pol = [chi2.cdf(np.shape(datos)[0] - (p + 2), X_sq_new[:][i]) for i in np.arange(0, int(n_draws))]

p_value_pol = np.ones(n_draws) - cdf_X_sq_pol

num_bins = 25
n, bins, patches = plt.hist(p_value_pol, num_bins, histtype='step')
plt.xlabel('$\mathrm{p-value}$')
plt.ylabel('$\mathrm{Frequency}$')
plt.show()
'''

plt.errorbar(datos[:, 0], datos[:, 1], yerr=datos[:, 2], ls='none', marker='p', label='$\mathrm{Original}$')
plt.plot(datos[:, 0], Y_theta)
plt.xlim(0.15, 0.85)
plt.ylim(0.9997, 1.0002)
plt.xlabel('$\mathrm{tiempo}$ $\mathrm{(dias)}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.show()
	
