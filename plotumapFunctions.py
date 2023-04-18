# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:44:20 2023

@author: Jenyi

Functions for plotting UMAP and TSNEs
"""

import os
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np

def plot_umap(data, labels, 
			  cell_names:list=None,
			  save_path:str='./'):
	

    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(data)
    num_clusters = np.unique(labels)
    if cell_names == None:
        cell_names = num_clusters
    temp = zip(num_clusters, cell_names)
    annotation_map = {k:v for k, v in temp}		
    for n in num_clusters:
        temp_trans = trans.embedding_[np.where(labels == n)[0], :]
        plt.scatter(temp_trans[:, 0], temp_trans[:, 1], s= 5, label=annotation_map[n])
    plt.title('UMAP of cell type clustering', fontsize=16)
    plt.legend(bbox_to_anchor=(1.2, 1.00))
    plt.savefig(os.path.join(save_path, 'umap.jpg'), dpi=60)
	
    return

