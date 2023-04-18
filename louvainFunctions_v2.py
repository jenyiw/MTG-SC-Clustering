# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:24:17 2023

@author: Jenyi
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq as hp
from scipy.sparse import csc_matrix
import plotumapFunctions as puF

def get_KNN(arr, k):

	"""
	Get k-NN for each point.
	Parameters:
		arr: (# of samples, # of features) numpy array
			 data array
	 k: int
			number of nearest neighbors for kNN graph

	 Returns:
		neighbor_arr: (# of samples, k+1) numpy array
		  Index of data points (1st column) and k-nearest neighbors
	 
	"""
	
	neighbor_arr = np.zeros((len(arr), k+1))
	
	for n in range(len(arr)):
		
		query_point = arr[n]
		
		dist_arr = list(np.sum((arr-query_point)**2, axis=1))
		
		heap_list = [(x, i) for (x, i) in zip(dist_arr, range(len(dist_arr)))]
		hp.heapify(heap_list)
		
		for m in range(k+1):
 			neighbor_arr[n, m] = hp.heappop(heap_list)[1]
		
	return neighbor_arr
		

def create_kNN_graph(arr, k):
	"""
	Construct kNN graph
	Parameters:
		arr: (# of samples, # of features) numpy array
			 data array
	 k: int
			number of nearest neighbors for kNN graph

	 Returns:
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix
	 
	"""
	
	edges = get_KNN(arr, k)
			
	edge_rows = []
	edge_cols = []	
	for e in range(len(edges)):
		for f in range(1,k+1):
			edge_rows.append(int(edges[e, 0]))
			edge_cols.append(int(edges[e, f]))
	
	edge_weights = np.ones_like(edge_rows)
	edge_mat = csc_matrix((edge_weights, (edge_rows, edge_cols)), shape=(len(arr), len(arr)), dtype=np.uint16)

	return edge_mat

def calculate_total_weight(edge_mat):
	
	"""
	Calcuate total weight of edges in the graph
	Parameters:
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix

	 Returns:
		total_weight: int
			Total weight of edges in the graph
	 
	"""
		
	total_weight = np.sum(np.triu(edge_mat.todense()))
	
	return total_weight


def calculate_delta_m(sum_tot_j,
					  kiin,
					  ki_m,):
	
	"""
	Calcuate delta Q
	Parameters:
		sum_tot_i: float
		  Sum of links to nodes in community
	  kiin: float
		  Sum of links from node i to nodes in community
	  ki_m: float
		  Normalized sum of links to i

	 Returns:
		delta_mm: float
		  delta Q
	 
	"""		
	delta_mm = kiin - sum_tot_j*ki_m
	
	return delta_mm	

def calculate_ki_m(edge_mat, cluster_arr):
	
	"""
	Calcuate sum of links to each point i
	Parameters:
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix
		cluster_arr: (# of nodes,) array of nodes
	  
	 Returns:
		ki_arr: (# of nodes,)
			Sum of links to each node
	 
	"""
	
	ki_arr = np.zeros_like(cluster_arr)
	
	for i in range(len(cluster_arr)):
		ki_arr[i] = np.sum(edge_mat[:, i])
	
	return ki_arr	

def calculate_kiin(i, c, edge_mat, cluster_arr):
	
	"""
	Calcuate sum of links to each point i
	Parameters:
		i: int
		 Index of node of interest
	   c: int
	    Index of community of interest
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix
		cluster_arr: (# of nodes,) array of nodes
	  
	 Returns:
		same_cluster: int
			Sum of links from i to nodes in c	
		sum_tot_j: int
			Sum of links to nodes in c
	 
	"""
	
	same_cluster = 0
	
	sum_tot_j = 0
	
	members = set(np.where(cluster_arr == c)[0])
	members.add(i)
	
	for n in members:
		if edge_mat[n, i] == 1:
			same_cluster += 1
		
	triangle_matrix = np.triu(edge_mat.todense())
	for n in members:
		sum_tot_j += np.sum(triangle_matrix[:, n])
		
	return same_cluster, sum_tot_j

def update_membership(membership_arr, cluster_arr, cluster_old):
	
	"""
	maps membership of each samples to each new class/node during each iteration
	Parameters:
		membership_arr: (# of samples,) numpy array
		    membership of each node
		cluster_old: (# of nodes,) numpy array
			Array of previous membership of each node
		cluster_arr: (# of nodes,)
			Array of new membership of each node	  
	 Returns:
		new_membership_arr: (# of samples,) numpy array
			Array with updated memberships	
	 
	"""	
	
	new_membership_arr = np.zeros((len(membership_arr)))
	
	for i in range(len(cluster_old)):
		
		#new cluster
		new_cluster = cluster_arr[i]
		
		#map all members in the node to new node index
		new_membership_arr[membership_arr == cluster_old[i]] = new_cluster

		
	return new_membership_arr

def reformat_cluster_arr(cluster_arr):
	
	"""
	Reset the clustering to start from 0
	Parameters:
		cluster_arr: (# of nodes,)
			Array of new membership of each node	  
	 Returns:
		cluster_arr: (# of nodes,) numpy array
			Array with updated cluster array
	 
	"""		
	
	clusters = list(np.unique(cluster_arr))
	cluster_arr = [clusters.index(x) for x in cluster_arr]
	cluster_arr = np.array(cluster_arr)
	
	return cluster_arr

def community_aggregation(edge_mat, cluster_arr):
	
	"""
	Community aggregation after each iteration.
	Parameters:
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix
		cluster_arr: (# of nodes,)
			Array of new membership of each node	  
	 Returns:
 		new_edge_mat: sparse matrix
		  New adjacency matrix as a sparse matrix
		new_cluster_arr: (# of new nodes,) numpy array
			Array with updated cluster array
	 
	"""	
	
	num_clusters = np.unique(cluster_arr)
	new_edge_mat = csc_matrix((len(num_clusters), len(num_clusters)), dtype=np.uint16)	
	
	for u in range(len(cluster_arr)):
		
	    for v in range(len(cluster_arr)):		
		    
		    if edge_mat[u, v] == 0:
		        continue
			
		    new_node_u = int(cluster_arr[u])
		    new_node_v = int(cluster_arr[v])
		
		    if new_node_u == new_node_v:
			
			    #add edge_weights to self_loop
			    new_edge_mat[new_node_u, new_node_u] += 1

		    #add edge_weights to edges between clusters				
		    elif new_node_u != new_node_v:
			
			    new_edge_mat[new_node_u, new_node_v] += 1						
			    new_edge_mat[new_node_v, new_node_u] += 1	
			
	new_cluster_arr = np.array(list(set(cluster_arr)))

	return new_edge_mat, new_cluster_arr				


def plot_graph(clusters_list, data_arr, membership_arr, edge_mat, t):
	
	
	"""
	PLot graph for louvain.
	
	Parameters:
		clusters_list: list
			list of clusters
		data_arr: (# of samples, # of features) numpy array
			 Data array
		edge_mat: sparse matrix
		  Adjacency matrix as a sparse matrix
		membership_arr: (# of samples,)
			Array of membership of each node
			t: int
			Iteration number  
	 Returns:
 		new_edge_mat: sparse matrix
		  New adjacency matrix as a sparse matrix
		new_cluster_arr: (# of new nodes,) numpy array
			Array with updated cluster array
	 
	"""		
	
	#generate random color palette
	colors = []
	for n in clusters_list:
		colors.append(np.random.rand(3,))
		
	#get corresponding color list
	clusters_list_temp = list(clusters_list)

	color_list = [colors[clusters_list_temp.index(j)] for j in membership_arr]

	edges = edge_mat.nonzero()

	for pt, pt2 in zip(edges[0], edges[1]):	
		plt.plot([data_arr[pt, 0], data_arr[pt2, 0]], [data_arr[pt, 1], data_arr[pt2, 1]], 'b-', markersize=0, alpha=0.7)
			
	plt.scatter(data_arr[:, 0], data_arr[:, 1], c=color_list, zorder=len(edges))
	for i, (x,y) in enumerate(data_arr):
	    plt.annotate(membership_arr[i], xy =(x, y))
	plt.title(f'Louvain clustering for iteration: {t}')
	plt.show()
	plt.close()
	

def create_trial_data(n):				
	#create fake data	
	data = np.random.normal(size=(n*2,))
	data_arr = np.zeros((n,2))
	data_arr[:, 0] = data[:n]
	data_arr[:, 1] = data[-n:]

	return data_arr

def louvain_clustering(data_arr,
					   k:int=2):
		
	"""
	Run Louvain
	Parameters:
		data_arr: (# of samples, # of features) numpy array
			 Data array
		 k: int
		   k for kNN graph
	 Returns:
 		membership_arr: (# of samples,) numpy array
			Array with membership of each node
	 
	"""	
	#create KNN graph
	edge_list = create_kNN_graph(data_arr, 2)
	edge_list_original = edge_list.copy()
	
	#initialize cluster array
	cluster_arr = np.zeros((len(data_arr)), dtype=np.uint16)
	cluster_arr[:] = range(len(data_arr))
	cluster_old = cluster_arr.copy()
	
	#initialize membership array
	membership_arr = np.zeros_like(cluster_arr)
	membership_arr[:] = range(len(cluster_arr))
	
# 	plot_graph(cluster_old, data_arr, membership_arr, edge_list, 0)
 	
	total_increase = 1
	t = 0
	
	print('Starting Louvain clustering')
	
	while total_increase > 0:
	
		#calculate total edges
		total_edges = calculate_total_weight(edge_list)
			
		#update modularity	
		total_increase = 0
		
		#calculate ki for all points
		ki_arr = calculate_ki_m(edge_list, cluster_arr)

		#random shuffle since order of iteration matters
		shuffled_index = np.arange(len(cluster_arr))
		np.random.shuffle(shuffled_index)
		
		for r in range(len(shuffled_index)):
			
			i = shuffled_index[r]
			
			#calculate ki
			ki = ki_arr[i]
			ki_m = ki/total_edges
			
			#store max values
			max_community = cluster_arr[i]
			max_increase = 0
			
			#try adding i to every cluster
			for c in cluster_arr:
				
				if c == cluster_arr[i]:
					continue
					
				#calculate kiin	and community sum sum_tot_j
				kiin, sum_tot_j = calculate_kiin(i, c, edge_list, cluster_arr)
				
				#calculate delta_m
				delta_m = calculate_delta_m(sum_tot_j, kiin, ki_m)
				
				#update if higher
				if delta_m > max_increase:
					max_increase = delta_m
					max_community = c
					cluster_arr[i] = c
				
			total_increase += max_increase
				
		theta = 0
		if total_increase < theta:
		 	break
		else:

		 	cluster_arr = reformat_cluster_arr(cluster_arr)	 
		 	membership_arr = update_membership(membership_arr, cluster_arr, cluster_old)			 
		 	edge_list, cluster_arr = community_aggregation(edge_list, cluster_arr)	  
		 	cluster_old = cluster_arr.copy()
		 
		t += 1
		print(f'Running iteration: {t}')
# 		plot_graph(cluster_old, data_arr, membership_arr, edge_list_original, t)
		
	return membership_arr
 
 
#function for testing against functions from packages 
# import matplotlib.cm as cm
# import networkx as nx
# import networkx.algorithms.community as nx_comm
# import community as community_louvain

# G=nx.Graph()
# G.add_edges_from(edge_list_original)

# #first compute the best partition
# partition = nx_comm.louvain_communities(G)
# print(partition)
# partition = community_louvain.best_partition(G)
# print(partition)
# # draw the graph
# pos = nx.spring_layout(G)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.show()

if __name__ == '__main__':
	
	data_arr = create_trial_data(15)
	membership_arr = louvain_clustering(data_arr, k=5)
	puF.plot_umap(data_arr, membership_arr)
	print(membership_arr)





		
	
	
	