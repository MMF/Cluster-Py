import numpy as np
from numpy.linalg import norm
import random

class MyKmeans:

    # constructor
    def __init__(self, n_clusters, max_iters=100, random_state=564, init_centroids=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

        # centroids
        self.centroids = init_centroids

    # find random centroid at the beginning
    def initialize_centroids(self, X):
        # set random state => guarantees that same randomization occurs every time
        np.random.RandomState(self.random_state)
        rand_idx = random.sample(range(X.shape[0]), self.n_clusters)
        return X[rand_idx]

    # compute centroids of data
    def compute_centroids(self, X, labels):
        # reset centroids
        # c1 = [0, 0]
        # c2 = [0, 0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)

        return centroids

    # compute distances to the centroids
    def compute_distance(self, X, centroids):
        # initialize distance vector
        # distance of every data point with each clustor_centroid
        distance = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            # compute Euclidean distance vector
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        return distance

    # return list of indices of  min values along  axis
    def find_closest_cluster(self, distance):
        # return indices of minimum values along axis
        return np.argmin(distance, axis=1)

    # sum of squared root of distances
    # @TODO: take min sum of a cluster
    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)

        return np.sum(np.square(distance))

    def fit(self, X):
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iters):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)

            # stopping condition (centroids didn't change)
            if np.all(old_centroids == self.centroids):
                break;

        self.error = self.compute_sse(X, self.labels, self.centroids)


    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)