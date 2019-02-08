#!/usr/bin/env python3

"""
This script shows different algorithms used for unregularized regression:
- Batch gradient descent:
      Gradient is computed from all samples at every iteration
- Stochastic gradient descent:
      Samples are submitted one at a time to compute gradient and update parameters
- Closed-form solution:
      When (X^T . X) is inversible, the closed-from solution allows to compute
      parameters directly with matrix calculus
"""

import numpy as np;
from numpy.linalg import inv;
import matplotlib.pyplot as plt;

def display(X, theta, label):
    """
    Computes and plots the output predicted from input X and parameters theta
    """
    plt.plot(X[:, 0], np.asarray(theta.T * X.T).T, label=label);


# Load dataset
FILENAME = "data_regression.txt";
data = np.loadtxt(fname = FILENAME);
X = np.ones((len(data), 2));
X[:, 0] = data[:, 0];
y = np.asmatrix(data[:, 1]).T;

# Batch gradient descent
alpha   = 0.0005;
nb_iter = 1000;
thetaBatch = np.matrix([0, 0]).T;
for iter in range(nb_iter):
    sum = np.matrix([0, 0]).T;
    for i in range(X.shape[0]):
        x = np.asmatrix(X[i]).T;
        sum = sum + (np.asscalar(thetaBatch.T * x - y[i, 0]) * x);
    thetaBatch = thetaBatch - alpha * sum;
print("Batch gradient descent");
print('theta=',thetaBatch,'\n');

# Stochastic gradient descent
thetaStoch = np.matrix([0, 0]).T;
alpha = 0.01;
nb_iter = 100;
for iter in range(nb_iter):
    for i in range(X.shape[0]):
        x = np.asmatrix(X[i]).T;
        thetaStoch = thetaStoch - alpha * (np.asscalar(thetaStoch.T * x - y[i, 0]) * x);
print("Stochastic gradient descent");
print('theta=',thetaStoch,'\n');

# Closed form solution
print("Closed form solution");
thetaClosed = np.dot( np.dot( np.linalg.inv(np.dot(X.T, X)), X.T), y);
print('theta=',thetaClosed,'\n');

# Display results
plt.plot(X[:, 0], y, label='Dataset');
display(X, thetaBatch,  'Batch gradient descent')
display(X, thetaStoch,  'Stochastic gradient descent')
display(X, thetaClosed, 'Closed-form solution')

plt.legend()
plt.show();
