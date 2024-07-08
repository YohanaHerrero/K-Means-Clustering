import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

df = pd.read_csv("Cust_Segmentation.csv")
df.head()

#for kmeans I can only use numerical data
df = df.drop('Address', axis=1)
df.head()

#normalize data set
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

#kmeans
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

#assign the labels to each row in the dataframe
df["Clus_km"] = labels
df.head(5)

#centroids (averaging)
df.groupby('Clus_km').mean()

#plot eg income vs age with the clusters found by kmeans
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

#plot in 3d age, income, education
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()

#k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. 
#The customers in each cluster are similar to each other demographically. Now we can create a profile for each group, 
#considering the common characteristics of each cluster. For example, the 3 clusters can be:
#AFFLUENT, EDUCATED AND OLD AGED
#MIDDLE AGED AND MIDDLE INCOME
#YOUNG AND LOW INCOME
