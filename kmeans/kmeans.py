#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

"""
This script demonstrates the execution of the k-means algorithm. It is a vector
quantization method used for unsupervised classification.
The goal here is to classify some 2D points coming from 4 different clusters.

Note: Parameter K here was set as equal to the number of clusters. Generally,
      this may not be a good practice and K should be much higher than the
      expected number of clusters to discriminate. Hence each expected cluster
      should be represented by several clusters computed from the algorithm.

      In some cases, the dataset is too complex to be able to expect a specific
      number of clusters. Some methods intend to determine the optimal K to choose,
      or some algorithms derived from k-means do not require to choose K beforehand
      as they automatically adjust the number of clusters during the execution.
"""

def fitSampleToPrototype(X, centroids):
    """
    Assigns each sample of X to the closest prototype/centroid according to
    the Euclidean distance
    Returns an array containing the index of the assigned prototype for each sample
    """
    minDist    = [np.inf] * len(X)
    minCluster = [0]      * len(X)
    for i in range(len(centroids)):
        dist = np.sqrt( np.power(X[:,0]-centroids[i,0],2) + np.power(X[:,1]-centroids[i,1],2) ).tolist()
        for sample in range(len(X)):
            if dist[sample] < minDist[sample]:
                minDist[sample]    = dist[sample]
                minCluster[sample] = i
    return minCluster

def fitPrototypesToSamples(X, clusters, K):
    """
    Computes the new centroids of the clusters as the averages of all the samples
    assigned to each of these clusters
    """
    centroids = np.asarray([[0.0,0.0]] * K)
    for c in range(K): # For each cluster
        fittedSamplesIdx = list(filter(lambda idx: clusters[idx]==c, range(len(X))))
        centroids[c,:] = np.average(X[fittedSamplesIdx,:], axis=0)
    return centroids

def displayClusters(X, centroids, clusters, colors, ax):
    """
    Plots:
    - the centroids of the cluster with a black square around them
    - the dataset with each sample colored the same as the associated centroid
    """
    ax.clear()
    ax.scatter(X[:,0], X[:,1], c=colors[clusters], marker='x', label='Dataset');
    ax.scatter(centroids[:,0], centroids[:,1], c=colors, marker='o', s=80, label='Cluster centroids', linewidths=2, edgecolors='black')
    ax.set_title('Dataset clustered with k-means')
    plt.pause(0.5)


N = 200     # Samples per cluster
K = 4       # Number of clusters

# Create dataset
means     = [(0,-2), (6,5), (-.7,5), (4,1.2)]
variances = [[2,2],  [3,1], [1.5,4], [0.5,2]]
colors    = np.asarray(['orange', 'royalblue', 'green', 'magenta'])
X         = np.zeros((N*len(means), 2))
y         = [0] * N*len(means)
for mean,var,i in zip(means, variances,range(len(means))):
    X[i*N:(i+1)*N, :] = np.random.multivariate_normal(mean, np.asarray(np.diagflat(var)), size=N)
    y[i*N:(i+1)*N]    = [i] * N

# Display dataset
f, axes = plt.subplots(1, 2, subplot_kw=dict(aspect='equal'))
axes[0].scatter(X[:,0], X[:,1], c=colors[y], marker='x', label='Dataset');
axes[0].set_title('Dataset with real clusters')

# Initialize clusters randomly among the dataset samples
centroids = X[np.random.randint(len(X), size=K), :]
clusters  = np.random.randint(K, size=len(y)).tolist()
displayClusters(X, centroids, clusters, colors, axes[1])

# K-means
convergence = False
nb_iter = 0
while not convergence:
    nb_iter += 1
    # Step 1 - assign samples to a prototype
    oldClusters = clusters
    clusters = fitSampleToPrototype(X, centroids)
    displayClusters(X, centroids, clusters, colors, axes[1])
    # Step 2 - update prototypes to the centroid of fitted samples
    centroids = fitPrototypesToSamples(X, clusters, K)
    # Display clusters and centroids
    displayClusters(X, centroids, clusters, colors, axes[1])
    # Check convergence
    convergence = (oldClusters==clusters)

print('---- K-means: Convergence reached after ', nb_iter, ' iterations ----')
plt.show()
