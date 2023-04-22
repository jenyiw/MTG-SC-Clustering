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


def split_data(label_arr,
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
    indices_arr = list(range(len(label_arr)))
    cluster_labels = np.unique(label_arr)
	
    train_points = []
	
    for n in range(cluster_labels):
        cluster_arr = indices_arr[cluster_labels == n]
        cluster_props = len(cluster_arr)/len(cluster_labels)
        chosen_points = list(np.random.choice(cluster_arr, size=int(cluster_props*train_proportion), replace=False))
        train_points = train_points + chosen_points
	
    train_labels = [label_arr[x] for x in train_points]
    test_points = [x for x in indices_arr not in train_points]
    test_labels = [label_arr[x] for x in test_points]	
	
    return np.array(train_points), np.array(train_labels), np.array(test_points), np.array(test_labels)
	

	 

