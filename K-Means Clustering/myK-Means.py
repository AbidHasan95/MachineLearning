#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:52:07 2018

@author: sohail
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#Finding optimal number of clusters by elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the mall dataset
kmeans = KMeans(n_clusters= 5, init= 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
y_kmeans = kmeans.fit_predict(X)

#Visualising clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1] , s= 10, c= 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1] , s= 10, c= 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1] , s= 10, c= 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1] , s= 10, c= 'brown', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1] , s= 10, c= 'cyan', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label= 'Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
