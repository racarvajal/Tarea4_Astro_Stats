#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use k-fold cross-validation to test degrees of polynomial part of model
# Select 'optimal' degree

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
p_max = 18  # maximum degree of polynomial
n_draws = 1e3
k_fold = 10  # Number of subsets to use in cross-validation
index_start = np.shape(datos)[0] / k_fold
delta_index = np.shape(datos)[0] / k_fold - 1
cdf_X_sq = np.zeros(p_max)
X_sq = np.zeros(p_max)

T = np.transpose(datos[:, 0])
Y = np.transpose(datos[:, 1])
L = np.transpose(np.ones(np.shape(datos)[0]))
V_inv = sigma**-2 * np.identity(np.shape(datos)[0])

indices = np.arange(0, np.shape(datos)[0])  # array of indices to filter validation set

MSE_local = np.zeros(k_fold)
MSE = np.zeros(p_max)

# Loop over parameter numbers
for p in np.arange(0, p_max):
	pars = p + 2
	
	# Loop over each subset
	for k in np.arange(0, k_fold):
		# Create filters to separate training and validation sets
		filter_validate = (indices >= (index_start * k)) & (indices <= (index_start * k + delta_index))
		filter_train = (indices < (index_start * k)) | (indices > (index_start * k + delta_index))

		# Separate time vector
		T_train = T[filter_train]
		T_validate = T[filter_validate]
	
		# Separate Y_k vector
		Y_train = Y[filter_train]
		Y_validate = Y[filter_validate]
	
		# Separate 1s vector
		L_train = L[filter_train]
		L_validate = L[filter_validate]
	
		# Separate variance-covariance matrix
		V_train_inv = sigma**-2 * np.identity(np.shape(T_train)[0])
		V_validate_inv = sigma**-2 * np.identity(np.shape(T_validate)[0])

		# Separate design matrix and fill it
		# Training design matrix
		M_train = np.zeros((np.shape(T_train)[0], pars))
		M_train[:, 0] = [-1 if (0.4 < t < 0.7) else 0 for t in T_train]
		M_train[:, 1:] = np.reshape([[np.power(T[j], i) for i in np.arange(0, pars - 1)] for j in np.arange(0, np.shape(T_train)[0])], (np.shape(T_train)[0], pars - 1))
	
		# Validation design matrix
		M_validate = np.zeros((np.shape(T_validate)[0], pars))
		M_validate[:, 0] = [-1 if (0.4 < t < 0.7) else 0 for t in T_validate]
		M_validate[:, 1:] = np.reshape([[np.power(T[j], i) for i in np.arange(0, pars - 1)] for j in np.arange(0, np.shape(T_validate)[0])], (np.shape(T_validate)[0], pars - 1))
	
		# Obtain parameters from training set with MLE
		# Theta = (Y_k - M.Theta - L)^t.V^{-1}.(Y_k - M.Theta - L)
		M_train_t = np.transpose(M_train)
		A_train = inv(np.dot(np.dot(M_train_t, V_train_inv), M_train))
		Y_L_train = Y_train - L_train
		B_train = np.dot(np.dot(M_train_t, V_train_inv), Y_L_train)

		Theta_train = np.dot(A_train, B_train)
		
		# Calculate Y for validation data: test model
		Y_validate_theta = np.dot(M_validate, Theta_train) + L_validate
	
		# Calculate Mean Square Error in validation set
		# MSE = sum((Y_validate - Y data)^{2}) / n
		MSE_local[k] = np.sum((Y_validate_theta - Y_validate)**2) / np.shape(Y_validate)[0]

	MSE[p] = MSE_local.mean()

# Plot K-fold Cross Validation in terms of MSE
plt.plot(np.arange(0, p_max), MSE, label=('$%i-\mathrm{fold}$' % k_fold), marker='p')
plt.xlabel('$\mathrm{p}$ : $\mathrm{parameters}$')
plt.ylabel('$\mathrm{MSE}$')
plt.legend(loc='best')
# plt.savefig('T4_p3_b.pdf')
plt.show()

