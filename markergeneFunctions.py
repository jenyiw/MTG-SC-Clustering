
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:28:39 2023

@author: Jenyi

Functions for extracting marker genes from clusters by pairwise comparison
"""
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

def get_gene_stats_by_cluster(data_path, 
					 cluster_labels, 
					 clusters):
	
	"""
	GEt statistics for each cluster and each gene.
	Parameters:
		data_path: str
			 path of the .h5 file
		 cluster_labels: numpy array
			 Cluster labels for each point
		 clusters: list
			 List of clusters
	 Returns:
		median_arr: (# of cluster, # of genes) numpy array
			   Array of median values for each cluster and each gene
	   mean_arr: (# of cluster, # of genes) numpy array
			   Array of mean values for each cluster and each gene		 
	
	
	"""

	with h5py.File(data_path, "r") as f:
		num_genes = f['data'].shape[1]
	median_arr = np.zeros((len(clusters), num_genes))
	mean_arr = np.zeros_like(median_arr)
# 	var_arr = np.zeros_like(mean_arr)
	with h5py.File(data_path, "r") as f:
	    for n in range(len(clusters)):
		
		    cluster_arr = f['data'][cluster_labels==clusters[n]]
		    mean_arr[n, :] = np.mean(cluster_arr, axis=0)
		    median_arr[n, :] = np.median(cluster_arr, axis=0)	
# 		var_arr[n, :] = np.var(cluster_arr, axis=0)
		
	return median_arr, mean_arr


def filter_genes_by_median(median_arr, mean_arr):	
	"""
	Get marker genes using the median expression >2 times other clusters and mean CPM counts > 1.
	
	Parameters:
		median_arr: (# of cluster, # of genes) numpy array
			   Array of median values for each cluster and each gene
	   mean_arr: (# of cluster, # of genes) numpy array
			   Array of mean values for each cluster and each gene
			   
   Returns:
	   None

	"""

	cluster_dict = {}
	marker_rank_dict = {}		
	for n in range(len(mean_arr)):
		cluster_dict[n] = []
		marker_rank_dict[n] = []		
	for n in range(mean_arr.shape[1]):
		top_cluster = np.argmax(median_arr[:, n])
		if mean_arr[top_cluster, n] >= 1:
		    median_exp = median_arr[top_cluster, n]
		    temp_median_arr = median_arr[:, n]
		    temp_median_arr = np.delete(temp_median_arr, top_cluster, axis=0)
		    if np.all(temp_median_arr*2 < median_exp):
		        cluster_dict[top_cluster].append(n)
		        marker_rank_dict[top_cluster].append(median_exp/np.mean(temp_median_arr))

	#sort markers by fold change over mean
	for n in cluster_dict.keys():
		a = cluster_dict[n]
		b = marker_rank_dict[n]
		sorted_genes = [x for _, x in sorted(zip(b, a))]
		sorted_fold_change = sorted(b)
		cluster_dict[n] = sorted_genes[::-1]
		marker_rank_dict[n] = sorted_fold_change[::-1]
				
	#save sorted arrays
	with open('marker_genes.json', 'w') as f:
	    json.dump(cluster_dict, f)
	with open('marker_genes_fc.json', 'w') as f:
	    json.dump(marker_rank_dict, f)		
		
	return

def plot_marker_genes(data_path, 
					  clusters, 
					  labels,
					  n_genes:int=3,
					  ):
	
	"""
	Plot marker genes as violin plots
	Parameters:
		data_path: str
			 path of the .h5 file
		 labels: numpy array
			 Cluster labels for each point
		 clusters: list
			 List of clusters
		 n_genes: int
			 number of genes to plot
	 Returns:
		None	 
	
	
	"""
	
	with open('marker_genes.json') as f:
		marker_genes = json.load(f)
	
	print(marker_genes)
	for n in range(len(clusters)):
		
		gene_list = marker_genes[str(n)]
		max_num = min(n_genes, len(gene_list))
		if max_num == 0:
			continue		
		fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
		fig.suptitle(f'Marker genes for cluster: {n}', fontsize=20)
		fig.supylabel(f'Gene expression', fontsize=20)
		
		with h5py.File(data_path, "r") as f:
		    for m in range(len(clusters)):
		        gene_matrix = f['data'][labels == clusters[m]]

		        for g in range(max_num):
			        gene = gene_list[g]
			        gene_data = gene_matrix[:, gene]
			        ax[g].violinplot(dataset=gene_data, positions=[m])
			        ax[g].set_xlabel(m, fontsize=16)
		
		    fig.tight_layout()
		    plt.savefig(f'marker_gene_{n}.jpg')
	
	
		
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:28:39 2023

@author: Jenyi

Functions for extracting marker genes from clusters by pairwise comparison
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def get_gene_stats_by_cluster(data_arr, 
					 cluster_labels, 
					 clusters):
	
	"""
	GEt statistics for each cluster and each gene.
	Parameters:
		data_arr: (# of samples, # of genes) numpy array
			 Array of data points
		 cluster_labels: numpy array
			 Cluster labels for each point
		 clusters: list
			 List of clusters
	 Returns:
		median_arr: (# of cluster, # of genes) numpy array
			   Array of median values for each cluster and each gene
	   mean_arr: (# of cluster, # of genes) numpy array
			   Array of mean values for each cluster and each gene		 
	
	
	"""

	median_arr = np.zeros((len(clusters), data_arr.shape[1]))
	mean_arr = np.zeros_like(median_arr)
# 	var_arr = np.zeros_like(mean_arr)
	for n in range(len(clusters)):
		
		cluster_arr = data_arr[cluster_labels==clusters[n]]
		mean_arr[n, :] = np.mean(cluster_arr, axis=0)
		median_arr[n, :] = np.median(cluster_arr, axis=0)	
# 		var_arr[n, :] = np.var(cluster_arr, axis=0)
		
	return median_arr, mean_arr


def filter_genes_by_median(median_arr, mean_arr):	
	"""
	Get marker genes using the median expression >2 times other clusters and mean CPM counts > 1.
	
	Parameters:
		median_arr: (# of cluster, # of genes) numpy array
			   Array of median values for each cluster and each gene
	   mean_arr: (# of cluster, # of genes) numpy array
			   Array of mean values for each cluster and each gene
			   
   Returns:
	   None

	"""

	cluster_dict = {}
	marker_rank_dict = {}		
	for n in range(len(mean_arr)):
		cluster_dict[n] = []
		marker_rank_dict[n] = []		
	for n in range(mean_arr.shape[1]):
		top_cluster = np.argmax(median_arr[:, n])
		if mean_arr[top_cluster, n] >= 1:
		    median_exp = median_arr[top_cluster, n]
		    temp_median_arr = median_arr[:, n]
		    temp_median_arr = np.delete(temp_median_arr, top_cluster, axis=0)
		    if np.all(temp_median_arr*2 < median_exp):
		        cluster_dict[top_cluster].append(n)
		        marker_rank_dict[top_cluster].append(median_exp/np.mean(temp_median_arr))

	#sort markers by fold change over mean
	for n in cluster_dict.keys():
		a = cluster_dict[n]
		b = marker_rank_dict[n]
		sorted_genes = [x for _, x in sorted(zip(b, a))]
		sorted_fold_change = sorted(b)
		cluster_dict[n] = sorted_genes[::-1]
		marker_rank_dict[n] = sorted_fold_change[::-1]
				
	#save sorted arrays
	with open('marker_genes.json', 'w') as f:
	    json.dump(cluster_dict, f)
	with open('marker_genes_fc.json', 'w') as f:
	    json.dump(marker_rank_dict, f)		
		
	return

def plot_marker_genes(data, 
					  clusters, 
					  labels,
					  n_genes:int=3,
					  ):
	
	with open('marker_genes.json') as f:
		marker_genes = json.load(f)
	
	print(marker_genes)
	for n in range(len(clusters)):
		
		gene_list = marker_genes[str(n)]
		max_num = min(n_genes, len(gene_list))
		if max_num == 0:
			continue		
		fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
		fig.suptitle(f'Marker genes for cluster: {n}', fontsize=20)
		fig.supylabel(f'Gene expression', fontsize=20)
		for m in range(len(clusters)):
		    gene_matrix = data[labels == clusters[m]]

		    for g in range(max_num):
			    gene = gene_list[g]
			    gene_data = gene_matrix[:, gene]
			    ax[g].violinplot(dataset=gene_data, positions=[m])
			    ax[g].set_xlabel(m, fontsize=16)
		
		fig.tight_layout()
		plt.savefig(f'marker_gene_{n}.jpg')
	
	