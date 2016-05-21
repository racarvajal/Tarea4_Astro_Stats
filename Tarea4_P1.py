#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Obtain parameters for transit model from data points
# Use MLE method with matrices
# Model will be Y_k = M.Theta + L

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
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

T = np.transpose(datos[:, 0])  # Time vector
Y = np.transpose(datos[:, 1])  # Y_k vector
L = np.transpose(np.ones(np.shape(datos)[0])) # 1s vector
V_inv = sigma**-2 * np.identity(np.shape(datos)[0])  # inverse of variance-covariance
M = np.zeros((np.shape(datos)[0], p + 2))  # Design matrix

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

# Obtain X^2 and p-value from model
# X^{2} = (Y_k - Y_theta)^t.V^{-1}.(Y_k - Y_theta)
C = (Y - Y_theta)
C_t = np.transpose(C)
X_sq = np.dot(np.dot(C_t, V_inv), C)

# P-value: P(chi_sq >= X^{2}) = 1 - P(chi_sq < X^{2})
cdf_X_sq = chi2.cdf(np.shape(datos)[0] - (p + 2), X_sq)
p_value = 1 - cdf_X_sq
print('The p-value is: %f' % p_value)
print('H_0 is rejected with a probability of: %f' % cdf_X_sq)
print('H_0 is rejected with a probability of: %f%%' % (cdf_X_sq * 100))

# Plot original data and model
plt.errorbar(datos[:, 0], datos[:, 1], yerr=datos[:, 2], ls='none', marker='p', label='$\mathrm{data}$')
plt.plot(datos[:, 0], Y_theta, label='$\mathrm{model}$')
plt.xlim(0.15, 0.85)
plt.ylim(0.9997, 1.0002)
ax = plt.gca()
ax.set_yticklabels(ax.get_yticks())
plt.xlabel('$\mathrm{tiempo}$ $\mathrm{(dias)}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc='best', title=('$\chi_{%i}^{2} = %.3f$\n$\mathrm{p-value}=%.3f$' % (np.shape(datos)[0] - (p + 2), X_sq, p_value)))
# plt.savefig('T4_p1.pdf')
plt.show()