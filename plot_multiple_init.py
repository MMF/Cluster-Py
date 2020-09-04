import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from  sklearn.datasets.samples_generator import (make_blobs, make_circles, make_moons)

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  silhouette_samples, silhouette_score
from My_KMeans import MyKmeans
import numpy as np

# ignore warning
from warnings import filterwarnings

filterwarnings('ignore')

# load data
data_path = 'D:\\Datasets\\geyser_eruption\\faithful.csv'
df = pd.read_csv(data_path)

# standardize data
stdScaler = StandardScaler()
data_std = stdScaler.fit_transform(df)

n_iters = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
# flatten array
ax = np.ravel(ax)
centers = []

for i in range(n_iters):
    # run KMeans
    km = MyKmeans(n_clusters=2,
                  max_iters=3,
                  random_state=np.random.randint(0, 1000, size=1))

    km.fit(data_std)
    centroids = km.centroids
    centers.append(centroids)

    ax[i].scatter(data_std[km.labels == 0, 0], data_std[km.labels == 0, 1],
                  c='green', label='cluster 1')
    ax[i].scatter(data_std[km.labels == 1, 0], data_std[km.labels == 1, 1],
                  c='blue', label='cluster 2')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='*', s=100, label='centroid')
    ax[i].set_xlim([-2, 2])
    ax[i].set_ylim([-2, 2])
    #ax[i].legend(loc='lower right')
    ax[i].set_title(f'{km.error:.4f}')
    ax[i].set_aspect('equal')

plt.subplots_adjust(hspace=0.5)
plt.show()