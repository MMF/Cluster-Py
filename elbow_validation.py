import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ignore warning
from warnings import filterwarnings

filterwarnings('ignore')

# load data
data_path = 'D:\\Datasets\\geyser_eruption\\faithful.csv'
df = pd.read_csv(data_path)

# standardize data
stdScaler = StandardScaler()
data_std = stdScaler.fit_transform(df)

sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(data_std)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot([_ for _ in range(1, 10)], sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.show()

