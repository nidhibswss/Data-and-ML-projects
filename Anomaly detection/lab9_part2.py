# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1) Load the data from the file 'dataOutliers.npy'


data2 = np.load('dataOutliers.npy')
print(data2.shape)

df = pd.DataFrame(data2, columns=['x', 'y'])

# 2) Create a scatter plot to visualize the data (This is just a FYI, make sure to comment the below line after you visualized the data)
#plt.scatter(x='x',y='y') 
#plt.title("data distribution2")
#plt.show()


# 3) Anomaly detection: Density-based
# Fit the LocalOutlierFactor model for outlier detection
# Then predict the outlier detection labels of the data points
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(df)



# 4) Plot results: make sure all plots/images are closed before running the below commands\
plt.close('all') 

# Create a scatter plot of the data (exact same as in 2) )
# Then, indicate which points are outliers by plotting circles around the outliers

x= df['x']
y = df['y']
plt.scatter(x,y)

# loop checking for outlier and making it red with black circle around it
for i in range(len(df)):
    if y_pred[i] == -1: # if i is outlier 
        plt.scatter(x[i], y[i], color='yellow', edgecolors='black')
plt.show()


