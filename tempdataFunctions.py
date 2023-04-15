# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:15:01 2023

@author: Jenyi

Create random data for code testing
"""
import numpy as np


def create_random_data(num_data,
					   n_clusters:int=5,
					   n_features:int=5):
	
    """
	Generate random data
	Parameters:
		num_data: int
		Number of data points required
		n_clusters: int
		Number of clusters required
		n_features: int
		Number of features required
	Returns:
		data_arr: (# of samples, # of features) numpy array
	    data_label: (# of samples) numpy array
		  Array with true labels
	
    """

    means = np.random.normal(size=(5,5))
    data_arr = np.zeros((num_data, n_features))
    l_cluster = num_data//n_clusters
    data_label = np.zeros((num_data, 1))
	
    for c in range(n_clusters):
        data_label[c*l_cluster:(c+1)*l_cluster] = c
        for f in range(n_features):
            data_arr[c*l_cluster:(c+1)*l_cluster, f] = np.random.normal(loc=means[c, f], scale=0.05, size=(l_cluster,))
			
    return data_arr, data_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_arr, data_label = create_random_data(100, n_features=2)

    plt.plot(data_arr[:, 0], data_arr[:, 1], 'b.')
    plt.show()



			

	
	
		
	
	