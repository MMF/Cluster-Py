# ignore warning
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
import random

from My_KMeans import MyKmeans

filterwarnings('ignore')

# number of clusters
n_clusters = 2

# load data
data_path = 'D:\\Datasets\\geyser_eruption\\faithful.csv'
df = pd.read_csv(data_path)

# standardize data
stdScaler = StandardScaler()
data_std = stdScaler.fit_transform(df)

minX = min(data_std[:, 0])
minY = min(data_std[:, 1])
maxX = max(data_std[:, 0])
maxY = max(data_std[:, 1])

lower_bounds = [minX, minY, minX, minY]
upper_bounds = [maxX, maxY, maxX, maxY]

"""
Objective function is sum of squared errors
total distance between points and related cluster_centroid
"""
def calc_fitness(sol):
    # solution represents centroids locations
    # sol[cent1_x, cent1_y, cent2_x, cent2_y]
    sol = np.array([sol[:2], sol[2:]])
    distances = compute_distance(sol)
    labels = np.argmin(distances, axis=1)
    sse = compute_sse(labels, sol)
    return sse

# compute distances to the centroids
def compute_distance(centroids):
    # initialize distance vector
    # distance of every data point with each clustor_centroid
    distance = np.zeros((data_std.shape[0], n_clusters))

    for k in range(n_clusters):
        # compute Euclidean distance vector
        row_norm = norm(data_std - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)

    return distance

# sum of squared root of distances
def compute_sse(labels, centroids):
    sse = np.zeros(data_std.shape[0])
    for k in range(n_clusters):
        sse[labels == k] = norm(data_std[labels == k] - centroids[k], axis=1)

    return np.sum(np.square(sse))