#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#read the csv
resample = pd.read_csv('together.csv')
resample.head()


# In[11]:


#Parameters
pp = resample[['GRID_CODE1-zoom','GRID_CODE-youtube','GRID_CODE-twitter','GRID_CODE-poi','GRID_CODE-flickr','GRID_CODE-email','GRID_CODE-company']]


# In[12]:


#Extra parameters
pp2=resample[['x','y']]


# In[13]:


#check for null values
pp.isnull().values.any()


# In[14]:


#gives no of null values
pp.isnull().sum().sum()


# In[15]:


sns.heatmap(pp.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# In[16]:


#Beginning of PCA

from sklearn.preprocessing import StandardScaler


# In[17]:


#scale, fit, transform
scaler = StandardScaler()  
scaler.fit(pp)
scaled_data=scaler.transform(pp)


# In[18]:


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_data)  #we fit the scaled data (above)
x_pca = pca.transform(scaled_data)  #this projects all our data to the 3 components


# In[19]:


scaled_data.shape


# In[20]:


x_pca.shape


# In[22]:


#Heatmap
sns.set(style="ticks", context="talk", font_scale=1, rc={"lines.linewidth": 1})
plt.style.use("dark_background")

pp_comp = pd.DataFrame(pca.components_, columns=pp.columns)
#ppnew = pp_comp.abs()
plt.figure(figsize=(24,12))

cmap = sns.diverging_palette(100,200, s=100,l=70,n=300, center="dark", as_cmap=True)
sns.heatmap(pp_comp, cmap=cmap, annot=True)
plt.savefig("PCA", dpi=300, bbox_inches = "tight")


# 
# ###### pp.corr()

# In[23]:


#Heatmap
sns.set(style="ticks", context="talk", font_scale=0.7, rc={"lines.linewidth": 1})
plt.style.use("dark_background")

plt.figure(figsize=(20,15))
heatmap= sns.heatmap(pp.corr(),cmap=cmap, annot=True)
plt.savefig("Correlation Heatmap", dpi=300, bbox_inches = "tight") #,transparent=True


# In[24]:


sns.set(style="ticks", context="talk", font_scale=1, rc={"lines.linewidth":0.1})
plt.style.use("dark_background")


plt.figure(figsize=(16,12))
plt.scatter(x_pca[:,0],x_pca[:,1],cmap='BuPu') #,c=resample['Crime']
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.savefig("PCA_graph", dpi=600)


# ### Simple pairplot with original data pp

# In[25]:


sns.set(style="ticks", context="talk", font_scale=1, rc={"lines.linewidth": 1})
plt.style.use("dark_background")

g = sns.PairGrid(pp)
g.map(plt.scatter)
plt.savefig("PairGrid", dpi=300, bbox_inches = "tight")


# ## K-Means Clustering
# 
# ### 1. Define number of clusters using Elbow method

# In[26]:


#Fit data and calculate sum of squares (WSS)
wss=[]
for i in range(1,12):
    from sklearn.cluster import KMeans
    kmeans_pca=KMeans(n_clusters=i,init="k-means++")
    kmeans_pca.fit(x_pca)
    wss.append(kmeans_pca.inertia_)


# In[27]:


#Visualisation of k values in order to define the fittest

sns.set(style="ticks", context="talk", font_scale=1, rc={"lines.linewidth": 1})
plt.style.use("dark_background")
plt.figure(figsize=(10,8))
plt.plot(range(1,12),wss,marker="o", color='#42c3a6')
plt.xlabel("Number of k clustering", color='white', fontsize='24') #,fontstyle='italic'
plt.ylabel("WSS", color='white', fontsize='24') #,fontstyle='italic'
#plt.yscale("log")

plt.savefig("Elbow Plot", dpi=300, bbox_inches = "tight")

plt.show()


# In[28]:


#pick number of cluster (manually count from elbow methood plot)
kmeans_pca=KMeans(n_clusters=4,init="k-means++")
#fit the data
kmeans_pca.fit(x_pca)


# In[29]:


kmeans_pca.labels_


# In[31]:


#adding out components columns to our dataframe
df=pd.DataFrame(x_pca,columns=["Component 1","Component 2","Component 3"])
data_pca_kmeans=pd.concat([resample,df],axis=1)
data_pca_kmeans["Segment kmeans PCA"]=kmeans_pca.labels_  # no need

#we add id,x, y to components and segment kmeans pca

data_pca_kmeans.to_csv('Data_PCA_kmeans.csv')

data_pca_kmeans.head()


# In[32]:


data_pca_kmeans["Segment"]=data_pca_kmeans["Segment kmeans PCA"].map({0:"first",1:"second",2:"third",3:"fourth",4:"fifth"})


# In[33]:


sns.set(style="ticks", context="talk", font_scale=1, rc={"lines.linewidth": 1})
plt.style.use("dark_background")

x_axis=data_pca_kmeans["Component 2"]
y_axis=data_pca_kmeans["Component 1"]
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue=data_pca_kmeans["Segment"],palette='BuPu')
plt.title("Clusters by PCA Components")

plt.savefig("Clusters by PCA Components", dpi=300) #,transparent=True

plt.show()


# In[34]:


#using data without PCA
kmeans=KMeans(n_clusters= 4, init="k-means++")
kmeans.fit(pp)
identified_clusters = kmeans.fit_predict(pp)
pp["Cluster_int"] =identified_clusters

pp.to_csv('Kmeans without PCA.csv')

pp.head()


# In[7]:


pwd


# In[ ]:




