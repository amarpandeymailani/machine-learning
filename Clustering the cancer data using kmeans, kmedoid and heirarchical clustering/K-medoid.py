#importing necessary libraries

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from copy import deepcopy

def euclidean_dist(data1, data2): #  calculating Eculdiean Distance between data points

    return np.sqrt(np.sum((data1 - data2)**2))

def initCentre(Cluster, Samples):  # It is randomly assigning centres to samples
    ids=[]
    while len(ids) < Cluster:
        n = np.random.randint(0,Samples)
        if not n in ids:
            ids.append(n)
    return ids


#implementation of k-medoids.

def kmedoid_algo(X, Cluster, dist_function, max_iter=400, tol=0.001, verbose=True):
    
    Samples, n_features = X.shape
    
    ids = initCentre(Cluster,Samples)

    centers = ids
    members, costs, total_cost, dist_mat = cost_eval(X, ids,dist_function)
    cc,SWAPED = 0, True
    while True:
        SWAPED = False
        for i in range(Samples):
            if not i in centers:
                for j in range(len(centers)):
                    centers_ = deepcopy(centers)
                    centers_[j] = i
                    members_, costs_, total_cost_, dist_mat_ = cost_eval(X, centers_,dist_function)
                    if total_cost_-total_cost < tol:
                        members, costs, total_cost, dist_mat = members_, costs_, total_cost_, dist_mat_
                        centers = centers_
                        SWAPED = True

        if not SWAPED:
            break
          
        cc =cc + 1
    return centers,members, costs, total_cost, dist_mat

def cost_eval(X, centers_id, dist_function):  #defining cost for datapoints
    
    dist_mat = np.zeros((len(X),len(centers_id)))
    
    for j in range(len(centers_id)):
        center = X[centers_id[j],:]
        for i in range(len(X)):
            if i == centers_id[j]:
                dist_mat[i,j] = 0.
            else:
                dist_mat[i,j] = dist_function(X[i,:], center)
    
    mask = np.argmin(dist_mat,axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))

    for i in range(len(centers_id)):  #assign centres to datapoints
        mem_id = np.where(mask==i)
        members[mem_id] = i
        costs[i] = np.sum(dist_mat[mem_id,i])
    return members, costs, np.sum(costs), dist_mat

class Kmedoid_fit(object):

    def __init__(self, Cluster, dist_function=euclidean_dist, max_iter=1000, tol=0.0001):  #feature initialization
        self.Cluster = Cluster
        self.dist_function = dist_function
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X,plotit=True, verbose=True): # It is assigning each point to one cluster and potting clusters
        centers,members, costs,total_cost, dist_mat = kmedoid_algo(
                X,self.Cluster, self.dist_function, max_iter=self.max_iter, tol=self.tol,verbose=verbose)
        if plotit:
            figure, ax = plt.subplots(1,1)
            colors = 10*['green','yellow']
            center_color = 10*['black','blue']
            if self.Cluster > len(colors):
                raise ValueError('we need more colors')
            
            cluster_1 = 0
            cluster_2 = 0
            
            for i in range(len(centers)):
                X_c = X[members==i,:]
                if i==0:
                    cluster_1=len(X_c)
                else:
                    cluster_2=len(X_c)
                ax.scatter(X_c[:,0],X_c[:,1],c=colors[i],s=25) # It is assigning color to each point in cluster 
                ax.scatter(X[centers[i],0],X[centers[i],1],alpha=0.9,c=center_color[i], s=100,marker='P')   # It is assigning color to each centroid in cluster
            

            
            print(" Number of data points cluster a " + str(cluster_1))
            print(" Number of data points cluster b " + str(cluster_2))
            plt.title('Clustering after applying K-Medoids with k=2')
            plt.xlabel('Radius_mean')
            plt.ylabel('Texture_mean')
            #plt.savefig('kmeadoid.png')
            plt.show()
       
        return

if __name__ == '__main__':
    
    data = pd.read_csv('cancer.csv')
    df = data.iloc[:,2:32]
    
    X = df.values 
    
    k_medoid = Kmedoid_fit(Cluster=2, dist_function=euclidean_dist)
    k_medoid.fit(X, plotit=True, verbose=True)
