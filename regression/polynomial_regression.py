#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

"""
This script shows regression with closed-form solution using polynomial models
of various orders. It shows the case of unregularized regression and also
RIDGE-regularized regression, which tends to minimize the magnitude of parameters
in order to add some bias and overall reduce the variance of the model.
"""

def h(theta, X):
    """
    Computes the outputs of the polynomial function which coefficients are in
    theta, for inputs X
    """
    return np.polynomial.polynomial.polyval(np.asarray(X), theta[::-1])[0]

def closedForm(X, y):
    """
    Closed form solution for unregularized regression
    Returns [ (X^T . X)^-1 ] . X^T . y
    """
    return np.dot( np.dot( np.linalg.inv(np.dot(X.T, X)), X.T), y)

def closedFormRidge(X, y):
    """
    Closed form solution for RIDGE-regularized regression
    Returns [ (X^T . X)^-1 + lambda.I ] . X^T . y
    """
    dim = X.shape[1]
    return np.dot( np.dot( np.linalg.inv(np.dot(X.T, X) + lambdaRidge*np.ones((dim,dim))), X.T), y)


def unregularized(X, y, order):
    """
    Computes theta parameters for unregularized regression with a polynomial
    model of specified order
    """
    X_train = np.ones((len(X), order+1))
    for i in range(order):
        X_train[:, i] = np.power(X, order-i).T
    return closedForm(X_train, y)

def regularizedRidge(X, y, order):
    """
    Computes theta parameters for RIDGE-regularized regression with a polynomial
    model of specified order
    """
    X_train = np.ones((len(X), order+1))
    for i in range(order):
        X_train[:, i] = np.power(X, order-i).T
    return closedFormRidge(X_train, y)


def meanSquareError(X, y, theta):
    """
    Computes Mean Square Error of model with theta perameters for input data X
    relatively to output data y
    """
    return np.sum(np.power(h(theta,X) - y, 2)) / len(X)

def display(X, y, thetas, colors, labels, title, ax):
    """
    Plots original output y and predicted output computed thanks to theta
    parameters
    """
    ax.plot(X, y, label='Dataset')
    for th,color,lab in zip(thetas,colors,labels):
        ax.plot(X, h(th,X), color, label=lab)
    ax.legend()
    ax.set_title(title)


TRAIN_PART = .7
lambdaRidge = 5

# Load dataset
FILENAME = "data_regression.txt"
data = np.loadtxt(fname = FILENAME)
X = np.asmatrix(data[:, 0]).T
y = np.asmatrix(data[:, 1]).T

# Separate train set from test set
X_train = X[0:int(len(X)*TRAIN_PART), :]
X_test  = X[len(X_train):len(X), :]
y_train = y[0:int(len(y)*TRAIN_PART), :]
y_test  = y[len(y_train):len(y), :]

# Unregularized linear
thetaLinear    = unregularized   (X_train, y_train, 1)
# Unregularized parabolic
thetaParabolic = unregularized   (X_train, y_train, 2)
# Unregularized 5-th order polynomial
theta5thOrder  = unregularized   (X_train, y_train, 5)
# RIDGE Regularized 5th order polynomial
theta5thRidge  = regularizedRidge(X_train, y_train, 5)

# Results
f, axes = plt.subplots(1, 2, sharey=True, sharex=True)
thetas = [thetaLinear, thetaParabolic, theta5thOrder, theta5thRidge]
colors = ['orange', 'green', 'magenta', 'black']
labels = ['Unregularized linear', 'Unregularized parabolic', 'Unregularized 5-th order polynomial', 'RIDGE 5-th order polynomial']
for X_set,y_set,txt,ax in zip([X_train, X_test], [y_train, y_test], ['Training set', 'Test set'], axes):
    print('===========', txt, '===========', '\n')
    # Print residual error
    for th,lab in zip(thetas, labels):
        print('----', lab, '----')
        print('Error = ', meanSquareError(X_set, y_set, th), '\n')
    # Display fitted curves
    display(X_set, y_set, thetas, colors, labels, txt, ax)

plt.show()
