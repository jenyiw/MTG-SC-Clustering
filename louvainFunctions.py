# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:24:17 2023

@author: Jenyi
"""
import numpy as np
from sklearn.neighbors import KDTree as ktree
import matplotlib.pyplot as plt


def create_kNN_graph(arr, k):
	
	tree = ktree(arr)
	dist, edges = tree.query(arr, k+1)
			
	edge_list = []
	for e in edges:
		for f in range(1,3):
			edge_list.append((int(e[0]),int(f)))
			
	for u, v in edge_list:
		if (v, u) in edge_list:
			edge_list.remove((v,u))
			
	edge_weights = np.ones((len(edge_list)))

	return edge_list, edge_weights

def calculate_delta_m(sum_tot_j,
					  kiin,
					  ki_m,):
		
	delta_mm = kiin - sum_tot_j*ki_m
	
	return delta_mm	

def calculate_ki_m(edge_list, edge_weights, cluster_arr):
	
	ki_arr = np.zeros_like(cluster_arr)
	for i, (u, v) in enumerate(edge_list):
		ki_arr[u] += edge_weights[i]
		ki_arr[v] += edge_weights[i]	
	
	return ki_arr	

def calculate_kiin(i, c, edge_list, edge_weights, cluster_arr):
	
	same_cluster = 0
	
	sum_tot_j = 0
	
	members = set(np.where(cluster_arr == c)[0])
	members.add(i)
	
	for j, (u, v) in enumerate(edge_list):
		
		if (u in members) or (v in members):
			sum_tot_j += edge_weights[j]
	
		if (u != i) and (v != i):
			continue
	
		if (u == i) and (cluster_arr[v] == c):
			same_cluster += edge_weights[j]	
			
		elif (v == i) and (cluster_arr[u] == c):
			same_cluster += edge_weights[j]	
		

	return same_cluster, sum_tot_j

def update_membership(membership_arr, cluster_arr, cluster_list):
	
	for i in range(len(cluster_list)):
		
		#new cluster
		new_cluster = cluster_arr[i]
		
		#map all members in the node to new node index
		membership_arr[membership_arr == cluster_list[i]] = new_cluster
		
	return membership_arr

def reformat_cluster_arr(cluster_arr):
	
	clusters = list(np.unique(cluster_arr))
	cluster_arr = [clusters.index(x) for x in cluster_arr]
	cluster_arr = np.array(cluster_arr)
	
	return cluster_arr

def community_aggregation(edge_list, edge_weights, cluster_arr):
	
	new_edge_list = []
	new_edge_weights = []
	
	
	for i, (u, v) in enumerate(edge_list):
		
		new_node_u = int(cluster_arr[u])
		new_node_v = int(cluster_arr[v])
		
# 		new_node_u = clusters.index(u_cluster)
# 		new_node_v = clusters.index(v_cluster)
		
		if new_node_u == new_node_v:
			
			#add edge_weights to self_loop
			if (new_node_u, new_node_u) not in new_edge_list:
				new_edge_list.append((new_node_u, new_node_u))
				new_edge_weights.append(edge_weights[i])
				
			else:
				index = new_edge_list.index((new_node_u, new_node_u))
				new_edge_weights[index] += 	edge_weights[i]	

		#add edge_weights to edges between clusters				
		elif new_node_u != new_node_v:
			
			if ((new_node_u, new_node_v) not in new_edge_list) and ((new_node_v, new_node_u) not in new_edge_list):
				
				new_edge_list.append((new_node_u, new_node_v))
				new_edge_weights.append(edge_weights[i])			
				
			else:
				if (new_node_u, new_node_v) in new_edge_list:
					index = new_edge_list.index((new_node_u, new_node_v))
				else:
					index = new_edge_list.index((new_node_v, new_node_u))					
				new_edge_weights[index] += 	edge_weights[i]			
	
	new_cluster_arr = np.array(list(set(cluster_arr)))
	
	new_edge_weights = np.array(new_edge_weights)

	return new_edge_list, new_edge_weights, new_cluster_arr				


def plot_graph(clusters_list, data_arr, membership_arr, edge_list, t):
	
	#generate random color palette
	colors = []
	for n in clusters_list:
		colors.append(np.random.rand(3,))
		
	#get corresponding color list
	clusters_list_temp = list(clusters_list)

	color_list = [colors[clusters_list_temp.index(j)] for j in membership_arr]

	for pt, pt2 in edge_list:	
		plt.plot([data_arr[pt, 0], data_arr[pt2, 0]], [data_arr[pt, 1], data_arr[pt2, 1]], 'b-', markersize=0, alpha=0.7)
			
	plt.scatter(data_arr[:, 0], data_arr[:, 1], c=color_list, zorder=len(edge_list))
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
	
	#create KNN graph
	edge_list, edge_weights = create_kNN_graph(data_arr, 2)
	edge_list_original = edge_list.copy()
	
	#initialize cluster array
	cluster_arr = np.zeros((len(data_arr)), dtype=np.uint16)
	cluster_arr[:] = range(len(data_arr))
	cluster_list = cluster_arr.copy()
	
	#initialize membership array
	membership_arr = np.zeros_like(cluster_arr)
	membership_arr[:] = range(len(cluster_arr))
	
	plot_graph(cluster_list, data_arr, membership_arr, edge_list, 0)
	
	total_increase = 1
	t = 0
	
	print('Starting louvain clustering')
	
	while total_increase > 0:
	
		#calculate total edges
		total_edges = np.sum(edge_weights)
			
		#update modularity	
		total_increase = 0
		
		#calculate ki for all points
		ki_arr = calculate_ki_m(edge_list, edge_weights, cluster_arr)
		
		#random shuffle since order of iteration matters
		shuffled_index = np.arange(len(cluster_arr))
		np.random.shuffle(shuffled_index)
		
		for r in range(len(shuffled_index)):
			
			i = shuffled_index[r]
			
			#calculate ki_m
			ki = ki_arr[i]
			ki_m = ki/(total_edges*2)
			
			#store max values
			max_community = cluster_arr[i]
			max_increase = 0
			
			#try adding i to every cluster
			for c in cluster_list:
				
				if c == cluster_arr[i]:
					continue
					
				#calculate kiin	and community sum sum_tot_j
				kiin, sum_tot_j = calculate_kiin(i, c, edge_list, edge_weights, cluster_arr)
				
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
		 	membership_arr = update_membership(membership_arr, cluster_arr, cluster_list)
		 	edge_list, edge_weights, cluster_arr = community_aggregation(edge_list, edge_weights, cluster_arr)	  
		 	cluster_list = cluster_arr.copy()
		 
		t += 1
		print(f'Running iteration: {t}')
		plot_graph(cluster_list, data_arr, membership_arr, edge_list_original, t)
		
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
	
	data_arr = create_trial_data(10)
	membership_arr = louvain_clustering(data_arr, k=2)
	print(membership_arr)





		
	
	
	