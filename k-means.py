#importing necessary libraries

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pandas as pd

#implementation of kmeans algorithm as Class and defining fit method

class Kmeans_algo:
    def __init__(self, k =2, tol = 0.001, max_iter = 400): # constructor to initialize features
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):        # method to create appropriate clusters using dataset

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]   #initializing centroids with starting 2 row values from dataset



        for i in range(self.max_iter):
            self.Class = {}
            for i in range(self.k):
                self.Class[i] = []

            for features in data:   #calculating l2 norms(eucl.dist) of difference between feature vector of a row and each entroid
                distances = [np.linalg.norm(features - self.centroids[CENTROID]) for CENTROID in self.centroids]
                Classification = distances.index(min(distances))
                self.Class[Classification].append(features)   #assigning data points to closest centroids




            prev = dict(self.centroids)
            #print(prev)   here you can see old selected centroids

            for Classification in self.Class:   #updating centroids by calculating mean of each Classes points
                self.centroids[Classification] = np.average(self.Class[Classification], axis = 0)

            isOptimal = True

            for CENTROID in self.centroids:      #comparing previous and curent centroids to know the converge

                original_CENTROID = prev[CENTROID]
                curr = self.centroids[CENTROID]

                if np.sum((curr - original_CENTROID)/original_CENTROID * 100.0) > self.tol:  #if error is greater than tolerance value then optimum cluster not found
                    isOptimal = False

            if isOptimal:  # if clusters are optimum stop the process
                break

        print(len(self.Class[0]))  #size of each cluster
        print(len(self.Class[1]))

    
if __name__ == "__main__":
    
    data = pd.read_csv('cancer.csv')   #loading dataset
    df = data.iloc[:,2:32]             #dropping first two non usable columns
    
    x = df.values
    k=2
    kmeans = Kmeans_algo(k)

    kmeans.fit(x)    #passing the array of dataset using k-means fuction

    center_color = 10*["blue", "green"]   # color to centroids
    Colors = 10*['yellow','red']         # color to datapoints
    
    cluster_1 = 0
    cluster_2 = 0
    
    # It is assigning color to each point with their respective centroid

    for Classification in kmeans.Class:
        color_assign = Colors[Classification]
        for features in kmeans.Class[Classification]:
            if color_assign == 'yellow':
                cluster_1 =cluster_1 + 1     #calculating cluster size by checking colors
            else:
                cluster_2 =cluster_2 + 1
            plt.scatter(features[0], features[1], color = color_assign,s = 25) # It is assigning color to each point in cluster
    
    # It assigns sign  and color to each centroid

    for CENTROID in kmeans.centroids:    #plotting figure between fisrt two features of dataset
        plt.scatter(kmeans.centroids[CENTROID][0], kmeans.centroids[CENTROID][1],c=center_color[CENTROID], s = 100, marker = "*") # It is assigning color to each centroid in cluster
   

            
    print("Number of data points in Cluster 1 : " + str(cluster_1))
    print("Number of data points in Cluster 2 : " + str(cluster_2))
    plt.title('Clusters with k-means | k=2')
    plt.xlabel('Radius_mean')
    plt.ylabel('Texture_mean')
    #plt.savefig('kmeans_fig.png')
    plt.show()
