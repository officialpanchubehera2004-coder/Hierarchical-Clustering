# Hierarchical Clustering


#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset= pd.read_csv(r"E:\FSDS_4pm\Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#scipy is an open source python libray which contain tools to do clusterning and build the dendogram. 
#we are not going to import whole scipy we are importing only scipy which related to the cluster and hierarachy


dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
#we are going to build the dendogram with only one line of code 
#linkage is one of the hieararchicla clustering algorithm & you have to build the linkage on X 
#ward method actually try to minimise the variance on each cluster & in k-means we minimise the sum of squared
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucliden distances')
plt.show()


# YOU CAN IMPLETE HEAR FIND ELBOW METOD 



# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
y_hc =hc.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# PLEASE COMPARE BOTH K-MEANS CLUSTERING vs HIERARCHICAL CLUSTERING

dataset['cluster']=y_hc
















