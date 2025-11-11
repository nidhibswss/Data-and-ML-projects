"""
Part 1
"""

# Imports
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt

"""
1.Form k clusters by assigning each instance to its nearest centroid.
2.Recompute the centroid of each cluster.
"""
# 1. Load the dataset (DataLab8.csv)
moviesDataset = pd.read_csv("DataLab8.csv")

# 2. To perform a k-means analysis on the dataset, extract only the numerical attributes: remove the "user" attribute 
data = moviesDataset.drop(columns=["user"])

## Suppose you want to determine the number of clusters k in the initial data 'data' ##
# 3. Create an empty list to store the SSE of each value of k (so that, eventually, we will be able to compute the optimum number of clusters k)
sse = []


# 4. Apply k-means with a varying number of clusters k and compute the corresponding sum of squared errors (SSE) 
# Hint1: use a loop to try different values of k. Think about the reasonable range of values k can take (for example, 0 is probably not a good idea).
# Hint2: research about cluster.KMeans and more specifically 'inertia_'
# Hint3: If you get an AttributeError: 'NoneType' object has no attribute 'split', consider downgrading numpy to 1.21.4 this way: pip install --upgrade numpy==1.21.4
k_range = range(1,6)

for k in k_range:
    kmeans = cluster.KMeans(n_clusters=k,random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)


#  5.  Plot to find the SSE vs the Number of Cluster to visually find the "elbow" that estimates the number of clusters. (read online about the "elbow method" for clustering)
plt.figure(figsize=(8,5))
plt.plot(k_range,sse,marker='x')
plt.title("k means clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 6. Look at the plot and determine the number of clusters k (read online about the "elbow method" for clustering)
k = 2

# 7. Using the optimized value for k, apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
kmeans_partition = cluster.KMeans(n_clusters=k,random_state=42)
kmeans_partition.fit(data)

labels = kmeans_partition.labels_

# Hint1: research about cluster.KMeans and more specifically 'labels_'


# 8. Display the assignments of each users to a cluster 
clusters = pd.DataFrame(labels, index=moviesDataset.user, columns=['Cluster ID'])
print(clusters)
