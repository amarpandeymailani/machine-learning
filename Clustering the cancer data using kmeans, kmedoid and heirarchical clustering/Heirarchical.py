import numpy as np                                                     #import library
import pandas as pd
import matplotlib.pyplot as plt

def euclid_dist(a,b):                                                          
    return np.sqrt(np.sum((a-b)**2))

class HeirarClus_avg:
    def __init__(self):                                                #constructor for class HeirarClus_avg no argument
        pass
    
    def clustering(self , x):
        self.x = x
        self.no_tup,self.no_features= x.shape                          #diamension of x in rows and columns
        self.distance=[]
      
        self.distance= np.zeros([self.no_tup, self.no_tup])
        b=[]
        for i in range(self.no_tup):                                   #formation of similarity matrix
            t=[]
            for j in range (self.no_tup):
                a= euclid_dist(self.x[i],self.x[j])
                t.append(a)
            b.append(t)
        for i in range(self.no_tup):
            for j in range (self.no_tup):
                if i!=j:
                    self.distance[i][j]=b[i][j]
                else:
                    self.distance[i][j]=9999999
        clust=[]
        for iter in range(self.no_tup-1):                               #iterating
            t=np.argmin(self.distance)
            row_no=t//self.no_tup
            col_no=t%self.no_tup
            b=[row_no,col_no]
            clust.append(b)
            self.update_matrix(self.distance,row_no,col_no)
        fin=self.create_cluster(clust)
        return self.cluster_no(fin)
    
        
    def update_matrix(self, m, r, c):                                  #updatation of similarity matrix
        for j in range (self.no_tup):
            if j!=r:
                m[j][r]=m[r][j]=(m[c][j]+m[r][j])/2
        for j in range (self.no_tup):
            m[j][c]=m[c][j]=9999999
       
    
    def create_cluster(self, c):                                       #creating cluster
        a=[[] for q in range(2)]
        for r in reversed(range(len(c))):
            if a[0]==[] and a[1]==[]:
                a[0].append(c[r][0])
                a[1].append(c[r][1])
            else:
                if c[r][0] in a[0]:
                    a[0].append(c[r][1])
                if c[r][0] in a[1]:
                    a[1].append(c[r][1])
        return a
    
    def cluster_no(self, clusters):                                      #assigning cluster no to clusters
        n= np.empty(self.no_tup)
        for i,j in enumerate(clusters):
            for l in j:
                n[l]=i
        return n
class HeirarClus_single:
    def __init__(self):                                                    #constructor for class HeirarClus_single no argument
        pass
    
    def clustering(self , x):
        self.x = x
        self.no_tup,self.no_features= x.shape                               #diamension of x in rows and columns
        self.distance=[]
      
        self.distance= np.zeros([self.no_tup, self.no_tup])
        b=[]
        for i in range(self.no_tup):                                        #formation of similarity matrix
            t=[]
            for j in range (self.no_tup):
                a= euclid_dist(self.x[i],self.x[j])
                t.append(a)
            b.append(t)
        for i in range(self.no_tup):
            for j in range (self.no_tup):
                if i!=j:
                    self.distance[i][j]=b[i][j]
                else:
                    self.distance[i][j]=9999999
        clust=[]
        for iter in range(self.no_tup-1):                                  #iterating
            t=np.argmin(self.distance)
            row_no=t//self.no_tup
            col_no=t%self.no_tup
            b=[row_no,col_no]
            clust.append(b)
            self.update_matrix(self.distance,row_no,col_no)
        fin=self.create_cluster(clust)
        return self.cluster_no(fin)
    
        
    def update_matrix(self, m, r, c):                                    #updatation of similarity matrix
        for j in range (self.no_tup):
            if j!=r:
                m[j][r]=m[r][j]=min(m[c][j],m[r][j])
        for j in range (self.no_tup):
            m[j][c]=m[c][j]=9999999
            
    def create_cluster(self, c):                                         #creating cluster
        a=[[] for q in range(2)]
        for r in reversed(range(len(c))):
            if a[0]==[] and a[1]==[]:
                a[0].append(c[r][0])
                a[1].append(c[r][1])
            else:
                if c[r][0] in a[0]:
                    a[0].append(c[r][1])
                if c[r][0] in a[1]:
                    a[1].append(c[r][1])
        return a
    
    def cluster_no(self, clusters):                                     #assigning cluster no to clusters
        n= np.empty(self.no_tup)
        for i,j in enumerate(clusters):
            for l in j:
                n[l]=i
        return n

    
class HeirarClus_complete:
    def __init__(self):                                                #constructor for class HeirarClus_single no argument
        pass
    
    def clustering(self , x):
        self.x = x
        self.no_tup,self.no_features= x.shape                          #diamension of x in rows and columns
        self.distance=[]
      
        self.distance= np.zeros([self.no_tup, self.no_tup])
        b=[]
        for i in range(self.no_tup):                                    #formation of similarity matrix
            t=[]
            for j in range (self.no_tup):
                a= euclid_dist(self.x[i],self.x[j])
                t.append(a)
            b.append(t)
        for i in range(self.no_tup):
            for j in range (self.no_tup):
                if i!=j:
                    self.distance[i][j]=b[i][j]
                else:
                    self.distance[i][j]=9999999
        clust=[]
        for iter in range(self.no_tup-1):                                   #iterating
            t=np.argmin(self.distance)
            row_no=t//self.no_tup
            col_no=t%self.no_tup
            b=[row_no,col_no]
            clust.append(b)
            self.update_matrix(self.distance,row_no,col_no)
        fin=self.create_cluster(clust)
        return self.cluster_no(fin)
    
        
    def update_matrix(self, m, r, c):                                       #updatation of similarity matrix
        for j in range (self.no_tup):
            if j!=r:
                m[j][r]=m[r][j]=max(m[c][j],m[r][j])
        for j in range (self.no_tup):
            m[j][c]=m[c][j]=9999999
            
    def create_cluster(self, c):                                            #creating cluster
        a=[[] for q in range(2)]
        for r in reversed(range(len(c))):
            if a[0]==[] and a[1]==[]:
                a[0].append(c[r][0])
                a[1].append(c[r][1])
            else:
                if c[r][0] in a[0]:
                    a[0].append(c[r][1])
                if c[r][0] in a[1]:
                    a[1].append(c[r][1])
        return a
    
    def cluster_no(self, clusters):                                       #assigning cluster no to clusters
        n= np.empty(self.no_tup)
        for i,j in enumerate(clusters):
            for l in j:
                n[l]=i
        return n
    
data = pd.read_csv('cancer.csv')                           #reading csv
data= data.to_numpy()
data= np.delete(data,[0,1,32],axis=1)
k= HeirarClus_avg()
pri=k.clustering(data)
q= HeirarClus_single()
qri=q.clustering(data)
z= HeirarClus_complete()
pri3=z.clustering(data)

print("Linkage= Single")
for i in range(len(set(qri))):
    print("Number of data points in cluster", i," is ", list(qri).count(i))


clr = ['red', 'green', 'blue','cyan','magenta', 'orange', 'brown', 'purple','olive','lime','black','aqua','teal','silver','peru']
X=data
y_km=qri
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
for i in range(len(set(y_km))):
    plt.scatter(
        X[y_km == i, 0], X[y_km == i, 1],
        s=50, c=clr[i%len(clr)],
        marker='o',label=i
    )


plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("Linkage= complete")
for i in range(len(set(pri3))):
    print("Number of data points in cluster", i," is ", list(pri3).count(i))

clr = ['red', 'green', 'blue','cyan','magenta', 'orange', 'brown', 'purple','olive','lime','black','aqua','teal','silver','peru']
X=data
y_km=pri3
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
for i in range(len(set(y_km))):
    plt.scatter(
        X[y_km == i, 0], X[y_km == i, 1],
        s=50, c=clr[i%len(clr)],
        marker='o',label=i
    )


plt.legend(scatterpoints=1)
plt.grid()
plt.show()

print("Linkage= Average")
for i in range(len(set(pri))):
    print("Number of data points in cluster", i," is ", list(pri).count(i))


clr = ['red', 'green', 'blue','cyan','magenta', 'orange', 'brown', 'purple','olive','lime','black','aqua','teal','silver','peru']
X=data
y_km=pri
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
for i in range(len(set(y_km))):
    plt.scatter(
        X[y_km == i, 0], X[y_km == i, 1],
        s=50, c=clr[i%len(clr)],
        marker='o',label=i
    )


plt.legend(scatterpoints=1)
plt.grid()
plt.show()

