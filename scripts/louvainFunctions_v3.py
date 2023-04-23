# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:24:17 2023

@author: Jenyi
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq as hp

import plotumapFunctions as puF
import networkx as nx

def get_KNN(arr, k):
	
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
	
	edges = get_KNN(arr, k)
			
	edge_list = []
	for e in range(len(edges)):
		for f in range(1,k+1):
			edge_list.append((int(edges[e, 0]),int(edges[e, f])))
			
	for u, v in edge_list:
		if (v, u) in edge_list:
			edge_list.remove((v,u))
			
	edge_weights = np.ones((len(edge_list)))

	return edge_list, edge_weights

def calculate_ki(G, i):
	
	
    neighbors = list(G.neighbors(i))
    ki = 0
    for n in neighbors:
        ki += G[i][n]['weight']
		
    return ki, neighbors
		

def calculate_kiin(i, c, G, cluster_arr):
	
	same_cluster = 0
	
	members = set(np.where(cluster_arr == c)[0])
	members.add(i)
	
	neighbor_nodes = nx.neighbors(G, i)
	for n in neighbor_nodes:
		if n in members:
			same_cluster += G[i][n]['weight']
	
	incident_edges = nx.edges(G, list(members))	
	sum_tot_j = 0
	
	for u, v in incident_edges:
		sum_tot_j += G[u][v]['weight']
		
	return same_cluster, sum_tot_j

def update_membership(membership_arr, cluster_arr, cluster_old):
	
	new_membership_arr = np.zeros((len(membership_arr)))
	
	for i in range(len(cluster_old)):
		
		#new cluster
		new_cluster = cluster_arr[i]
		
		#map all members in the node to new node index
		new_membership_arr[membership_arr == cluster_old[i]] = new_cluster

		
	return new_membership_arr

def reformat_cluster_arr(cluster_arr):
	
	clusters = list(np.unique(cluster_arr))
	cluster_arr = [clusters.index(x) for x in cluster_arr]
	cluster_arr = np.array(cluster_arr)
	
	return cluster_arr

def community_aggregation(G, cluster_arr):
	
	
	new_G=nx.Graph()
	
	for i, (u, v) in enumerate(G.edges):
		
		new_node_u = int(cluster_arr[u])
		new_node_v = int(cluster_arr[v])
		
		if new_node_u == new_node_v:
			
			#add edge_weights to self_loop
			if new_G.has_edge(new_node_u, new_node_u):
				new_G[new_node_u][new_node_u]['weight'] += G[u][v]['weight']
				
			else:
				new_G.add_edge(new_node_u, new_node_u, weight = G[u][v]['weight'])
				

		#add edge_weights to edges between clusters				
		else:
			
			if new_G.has_edge(new_node_u, new_node_v):
				new_G[new_node_u][new_node_v]['weight'] += G[u][v]['weight']			

			else:
				new_G.add_edge(new_node_u, new_node_v, weight = G[u][v]['weight'])						
	
	new_cluster_arr = np.array(list(set(cluster_arr)))

	return new_G, new_cluster_arr				


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
	
	#create KNN graph
	edge_list, edge_weights = create_kNN_graph(data_arr, 2)
	edge_list_original = edge_list.copy()
	G=nx.Graph()
	G.add_edges_from(edge_list, weight=1)
	
	#calculate total weight
	total_edges = 0
	for u, v in G.edges:
		   total_edges += G[u][v]['weight']
	
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

        #reset delta tracker
		total_increase = 0		

		#random shuffle since order of iteration matters
		shuffled_index = np.arange(len(cluster_arr))
		np.random.shuffle(shuffled_index)
		
		for r in range(len(shuffled_index)):
			
			i = shuffled_index[r]
			
			#calculate ki
			
			ki, neighbors = calculate_ki(G, i)
			ki_m = ki/total_edges
			
			#store max values
			max_community = cluster_arr[i]
			max_increase = 0
			
			possible_c = [cluster_arr[c] for c in neighbors]
			
			#try adding i to every cluster
			for c in possible_c:
				
				if c == cluster_arr[i]:
					continue
					
				#calculate kiin	and community sum sum_tot_j
				
				kiin, sum_tot_j = calculate_kiin(i, c, G, cluster_arr)
				
				#calculate delta_m
				delta_m = kiin - sum_tot_j*ki_m
				
				#update if higher
				if delta_m > max_increase:
					max_increase = delta_m
					max_community = c
					cluster_arr[i] = c
				
			total_increase += max_increase
				
		theta = 1e-07
		if total_increase < theta:
		 	break
		else:

		 	cluster_arr = reformat_cluster_arr(cluster_arr)	 
		 	membership_arr = update_membership(membership_arr, cluster_arr, cluster_old)			 
		 	G, cluster_arr = community_aggregation(G, cluster_arr)	
		 	cluster_old = cluster_arr.copy()
		 
		t += 1
# 		print(f'Running iteration: {t}')
# 		plot_graph(cluster_old, data_arr, membership_arr, edge_list_original, t)
	
	
	return membership_arr
 
 
#function for testing against functions from packages 
import matplotlib.cm as cm

import networkx.algorithms.community as nx_comm
import community as community_louvain



if __name__ == '__main__':
	
# 	data_arr = create_trial_data(15)
 	
#  	edge_list_original, edge_weights = create_kNN_graph(data_arr, 5)
#  	G=nx.Graph()
#  	G.add_edges_from(edge_list_original)

#  	#first compute the best partition
#  	partition = nx_comm.louvain_communities(G)
#  	print(partition)
#  	partition = community_louvain.best_partition(G)
#  	print(partition)
#  	# draw the graph
#  	pos = nx.spring_layout(G)
#  	# color the nodes according to their partition
#  	cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
#  	nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#  	                        cmap=cmap, node_color=list(partition.values()))
#  	nx.draw_networkx_edges(G, pos, alpha=0.5)
# 	plt.show()
	
	import os
	os.chdir('..')
	data_arr = np.loadtxt(os.path.join(r'./data', 'reduced_sample_20_PCs.csv'), delimiter=',')
	membership_arr = louvain_clustering(data_arr, k=5)
	puF.plot_umap(data_arr, membership_arr)
	print(np.unique(membership_arr))





		
	
	
	