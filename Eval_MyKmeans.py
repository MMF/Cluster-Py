from My_KMeans import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1

km = MyKmeans(n_clusters=2)
km.fit(X)

print(km.centroids)

km2 = KMeans(n_clusters=2, init='random')
km2.fit(X)
print("---------------------------")
print("skLearn KMeans")
print(km2.cluster_centers_)

plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(km.centroids[0, 0], km.centroids[0, 1], s=50, c='g', marker='s')
plt.scatter(km.centroids[1, 0] ,  km.centroids[1, 1], s=50, c='r', marker='s')
plt.show()
