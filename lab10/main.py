import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Country-data.csv')
print(dataset)

#K-means

X = dataset.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = '#9cb19a', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'rosybrown', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = '#8ba5c0', label = 'Cluster 3')

plt.scatter(X[1, 0], X[1, 1], s = 100, c = 'brown', label = 'Point')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = '#fce4bb', label = 'Centroids')
plt.title('Clusters of countries')
plt.xlabel('Exports')
plt.ylabel('Health')
plt.legend()
plt.show()

#Hierarchical

from scipy.cluster.hierarchy import linkage, dendrogram

mergings = linkage(X, method='ward')
dendrogram(mergings)

plt.title('Dendrogram')
plt.xlabel('Countries')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = '#9cb19a', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'rosybrown', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = '#8ba5c0', label = 'Cluster 3')
plt.scatter(X[1, 0], X[4, 1], s = 100, c = 'brown', label = 'Point')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = '#fce4bb', label = 'Centroids')
plt.title('Clusters of countries')
plt.xlabel('Exports')
plt.ylabel('Health')
plt.legend()
plt.show()