#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[24]:




'''
Step 2: Mean-centre every feature and scale the data by standard deviation of that feature
'''


def scale(data):
    mean = np.mean(data, axis=0)  # take mean of each column
    sd = np.std(data, axis=0)  # take standard deviation of each column
    mean_subtracted = data - mean
    sd[sd == 0] = 0.0000001
    scaled_data = mean_subtracted / sd
    return scaled_data


'''
Step 3: Log-transform the data
'''


def log_transform(scaled_data):
    return np.log(scaled_data)


'''
Step 4: Use PCA to keep features that capture 95% variance
'''

def perform_pca(data, n_components=20):
    model = PCA(n_components=n_components)
    model.fit(data)
    data = model.transform(data)
    return model.explained_variance_ratio_, data


'''
Step 5: Visualize the data
'''

# Plot the PCs against variance captured


def plot_PCs(variances):
    variances = variances * 100
    plt.bar(x=[str(i) for i in range(len(variances))], height=variances)
    plt.title("Principal Components")
    plt.xlabel("Component")
    plt.ylabel("Variance (%)")
    plt.show()


# In[3]:


data = np.loadtxt("../output/data_loaded.csv", dtype=float, delimiter=",")
data[data == 0.0] = 0.0000001
print("Data is loaded. Now taking log-transform. ")


# In[4]:


data = log_transform(data)
print("Data is transformed. Now scaling.")


# In[25]:


data = scale(data)
print("Data is mean-centred. Now performing PCA.")


# In[26]:


variances, reduced_data = perform_pca(data)
print("PCA successful! Now saving data.")


# In[27]:


sample = reduced_data[:500, :]
np.savetxt("../output/reduced_sample_20_PCs.csv", sample, delimiter=",", fmt="%.2f")


# In[28]:


np.savetxt("../output/reduced_data_20_PCs.csv", reduced_data, delimiter=",", fmt="%.2f")
print("Data saved! Now plotting.")


# In[29]:


plot_PCs(variances)


# In[30]:


sum(variances)


# In[ ]:




