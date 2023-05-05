# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:40:39 2023

@author: Jenyi

Code for managing data for learning models
"""

import os
import re
import pandas as pd
import numpy as np


def split_data(data_arr,
		       label_arr,
			   train_proportion:float=0.5,
			   ):
	
    """
	Split the data into training and testing datasets
	Parameters:
	   label_arr: numpy array
		   Matrix of (# samples,) of predicted labels
		train_proportion: float
			Proportion of whole dataset to use for training and not testing

	Returns:
		train_points: np.array
			indices of training_data
		train_label: np.array
			labels of training data
		test_points: np.array
			indices of test data
		test_label: np.array
			labels of test data
    """
    indices_arr = np.array(range(len(label_arr)))
    cluster_labels, counts = np.unique(label_arr, return_counts = True)
    mean_counts = np.mean(counts)
    num_test = mean_counts*(1-train_proportion)
    num_train = mean_counts*train_proportion
	
    train_points = []
    test_points = []
    train_labels = []
    test_labels = []
	

    for n in range(len(cluster_labels)):

        cluster_arr = indices_arr[label_arr == cluster_labels[n]]
        if len(cluster_arr) < 10:
            print('skip cluster', n)
        cluster_props = len(cluster_arr)/len(cluster_labels)
        chosen_points = list(np.random.choice(cluster_arr, size=int(cluster_props*train_proportion), replace=False))	
		
        test = [x for x in cluster_arr if x not in chosen_points] 
        if len(test) >= mean_counts:
            replace_mode = False
        else:
            replace_mode = True   		 
        chosen_test = list(np.random.choice(test, size=int(num_test), replace=replace_mode))
        test_points.append(chosen_test)
        test_labels.append([n]*len(chosen_test))		
        
        if len(chosen_points) >= mean_counts:
            replace_mode = False
        else:
            replace_mode = True   
        chosen_train = list(np.random.choice(test, size=int(num_train), replace=replace_mode))	 
        train_points.append(chosen_train)
        train_labels.append([n]*len(chosen_train))		
		
	
    train_labels = np.concatenate(train_labels)
    train_points = np.concatenate(train_points)
    train_data = data_arr[train_points]
		
    test_labels = np.concatenate(test_labels)
    test_points = np.concatenate(test_points)
    test_data = data_arr[test_points]	
	

    return train_data, train_labels, test_data, test_labels
	

	 

