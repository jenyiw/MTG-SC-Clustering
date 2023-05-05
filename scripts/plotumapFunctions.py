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
import matplotlib.colors as mcolors

def plot_umap(data, 
			  labels, 
			  title_list,
			  save_path:str='./'):
	
    """
	For plotting UMAP.
	Parameters:
		data: data matrix of (# samples, # genes)
		labels: cell label matrix of (# samples) containing cluster labels of each cell
		title_list: list with title for each plot
		save_path: str path to save the UMAP to
		
	Returns:
		None
	
    """

    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(data)
    color_list = list(mcolors.CSS4_COLORS.keys())
	
    fig, ax = plt.subplots(1, len(labels), figsize=(10,5))
    for i, l_arr in enumerate(labels):
        num_clusters = np.unique(l_arr)
        temp = zip(num_clusters, num_clusters)
        annotation_map = {k:v for k, v in temp}        		
        for n in num_clusters:
            temp_trans = trans.embedding_[np.where(l_arr == n)[0], :]
            ax[i].scatter(temp_trans[:, 0], temp_trans[:, 1], s= 5, color=color_list[int(n)], label=annotation_map[n])
            ax[i].axis('off')
            ax[i].set_title(title_list[i])
			
    plt.legend(bbox_to_anchor=(1.2, 1.00))

    plt.suptitle('UMAP of cell type clustering', fontsize=16)   
    plt.savefig(os.path.join(save_path, 'umap.jpg'), dpi=60)
	
    return

