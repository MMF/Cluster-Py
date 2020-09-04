# ignore warning
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from My_KMeans import MyKmeans
import numpy as np

filterwarnings('ignore')

# load data
data_path = 'dataset\\faithful.csv'
df = pd.read_csv(data_path)

# standardize data
stdScaler = StandardScaler()
data_std = stdScaler.fit_transform(df)

# start clustering
i_centroid = [
    [-1.26008544, -1.20156746],
    [0.70970326, 0.67674487]
]

# **********************************************************************************
km = MyKmeans(n_clusters=2, max_iters=100, init_centroids=np.array(i_centroid))
km.fit(data_std)

centroids = km.centroids
# -0.55038212   -0.52482256
print("[My KMeans] Centroids")
print(centroids)
print("Sum Squared Errors = " + str(km.error))
print("=================================================================")
# **********************************************************************************
km2 = KMeans(n_clusters=2, max_iter=100)
km2.fit(data_std)
print("Sk-Learn KMeans")
print("Centroids")
print(km2.cluster_centers_)
print("------------------")
print("Sum Squared Errors = " + str(km2.inertia_))


# plot centroids
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(data_std[km.labels == 0, 0], data_std[km.labels == 0, 1], c='green', label='cluster 1')
plt.scatter(data_std[km.labels == 1, 0], data_std[km.labels == 1, 1], c='blue', label='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r', label='centroid')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal');

plt.show()

























