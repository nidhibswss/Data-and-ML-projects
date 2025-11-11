# -*- coding: utf-8 -*-
"""
Part 2
Using the data from the file ‘vertebrate.csv’, we apply the following hierarchical clustering methods:
1. Single link (MIN)
2. Complete link (MAX)
3. Group average
"""

import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

#  Import the vertebrate.csv data
data = pd.read_csv("vertebrate.csv")

# Pre-process data: create a new variable and bind it with all the numerical attributes (i.e. all except the 'Name' and 'Class')
columns=['Name','Class']
numerical_attributes = data.drop(columns=columns)


### Single link (MIN) analysis + plot associated dendrogram ###
min_analysis = hierarchy.linkage(numerical_attributes,method='single')


# Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(10,6))

hierarchy.dendrogram(
    min_analysis,
    labels=data['Name'].tolist(),
    orientation='right'
    )

plt.title("single link MIN dendrogram")
plt.show()

### Complete Link (MAX) analysis + plot associated dendrogram ###
max_analysis = hierarchy.linkage(numerical_attributes,method='complete')

# Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(10,6))

hierarchy.dendrogram(
    max_analysis,
    labels=data['Name'].tolist(),
    orientation='right'
    )

plt.title("complete link MAX dendrogram")
plt.show()

### Group Average analysis ###
average_analysis = hierarchy.linkage(numerical_attributes,method='average')

#  Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
plt.figure(figsize=(10,8))

hierarchy.dendrogram(
    average_analysis,
    labels=data['Name'].tolist(),
    orientation='right'
    )

plt.title("group average dendrogram")
plt.show()
