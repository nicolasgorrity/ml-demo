#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

"""
This script shows the result of Principal Component Analysis (PCA) on a dataset
made of 2D points. It displays the resulting data with the principal eigenvectors.

PCA consists in projecting data into dimensions that are linear combinations
of initial dimensions, so the first dimensions gather as much data variance as
possible. Then the dimensions with the less amount of variance are ignored.

The purpose of PCA is to project complex data into a lower dimension subspace.
This can be useful to avoid overfitting.

PCA function was implemented so it can be used in two different ways:
- The user can require a specific dimension for the resulting subspace.
- The user can set a minimum proportion of variance that he desires to preserve
  in the data, so the dimension of the resulting subspace will be automatically
  chosen.
"""

def display(X, color, label):
    """
    Plots data X in specified color
    """
    plt.scatter(X[:,0], X[:,1], c=color, marker='x', label=label);

def rotate(origin, point, angle):
    """
    Rotates a point counterclockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    ox, oy = origin[0],  origin[1]
    px, py = point[:,0], point[:,1]
    an     = math.radians(angle)
    qx = ox + math.cos(an) * (px - ox) - math.sin(an) * (py - oy)
    qy = oy + math.sin(an) * (px - ox) + math.cos(an) * (py - oy)
    return np.asarray([qx, qy]).T

def mean_data(data):
    """
    Computes the mean value of the samples contained in dataset `data`.
    `data` should be a (N x d) matrix, with N the number of samples and
     d the dimension of each sample
    """
    return np.mean(data, axis=0)

def covariance(data):
    """
    Computes the covariance matrix of input data matrix
    If data is a (N x d) matrix, with N the number of samples and d the
    dimension of each sample, the returned covariance matrix size will
    be (d x d)
    """
    return np.dot(data.T, data) / len(data)
    # return np.cov(data.T)

def PCA(X, dim=None, min_var=None):
    """
    Executes Principal Components Analysis onto X data.
    If `dim` is not None, data will be projected into a subspace of given dimension
    If `dim` is None, the subspace dimension will be chosen as the minimum dimension
    so that `min_var` is the minimal proportion of variance that will be preserved.
    If none of `dim` or `min_var` are specified, an exception will be raised.
    `dim` should be a positive integer and `min_var` a float number in ]0, 1]
    """
    if dim is None and min_var is None:
        raise Exception('Either dim or min_var argument should be specified')
    # Compute data covariance matrix
    cov = covariance(X - mean_data(X))
    # Find eigenvalues with associated eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov)
    eig_vals = np.abs(eig_vals)
    # Sort them in decreasing order of eigenvalues
    eig_pairs = sorted([(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))], reverse=True)
    # Select K features so `var`% variance of data is maintained
    total = sum(eig_vals)
    eig_var = [pair[0]/total for pair in eig_pairs]
    cum_eig_var = np.cumsum(eig_var)
    if dim is not None and dim>0:
        K = int(dim)
    else:
        K = min(len(eig_vals), 1 + len(list(filter(lambda v: v < min_var, cum_eig_var))))
    # Compute projection matrix made of K first eigenvectors
    M = np.asarray([pair[1] for pair in eig_pairs[:K]]).T
    # Return data
    return M, K, cum_eig_var[K-1], [pair[0] for pair in eig_pairs[:K]]


# Samples in dataset
N = 100

# Create dataset
mean     = (2.5,4)
variance = np.asarray(np.diagflat([0.9,6]))
angle    = 40
X        = np.zeros((N, 2))
data     = np.random.multivariate_normal(mean, variance, size=N)
X        = rotate(mean, data, angle)


# Project data into optimized subspace
M, K, var, eigval = PCA(X, min_var=0.8)
X_mean = mean_data(X)
X_pca  = np.dot(X - X_mean, M)

# Reconstruct data
X_reconst = np.dot(X_pca, M.T) + X_mean


# Display original data
display(X, 'royalblue', 'Original dataset')
# Display reconstructed data
display(X_reconst, 'orange', 'Reconstructed data')
# Display principal eigenvectors
colors = np.asarray(['red', 'blue'])
if max(eigval)>4: eigval = np.asarray(eigval) * 4. / max(eigval)
plt.quiver([X_mean[0]], [X_mean[1]], [v*m for v,m in zip(eigval, M[0,:])], [v*m for v,m in zip(eigval, M[1,:])], color=colors, units='xy', scale=1., scale_units='xy')
# Show graph
plt.title('PCA: '+ str(K) +'-dimension subspace, '+ str(int(100*var)) +'% data variance preserved')
plt.axis('equal')
plt.legend()
plt.show()
