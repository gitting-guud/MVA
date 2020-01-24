"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    L = nx.laplacian_matrix(G)
    L = L.astype(float)
    eig_values, eig_vectors = eigs(L, k=100, which="SR")
    eig_vectors = eig_vectors.real
    
    km = KMeans(n_clusters=k)
    km.fit(eig_vectors)
    
    clustering = dict()
    for i, node in enumerate(G.nodes()):
        clustering[node] = km.labels_[i]
    return clustering



############## Task 6
G=nx.read_edgelist("D:\MVA\Altegrad\Lab_5\code/datasets/CA-HepTh.txt",delimiter = "\t")
gcc = max(nx.connected_component_subgraphs(G), key=len)
clustering = spectral_clustering(gcc, 50)

############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    modularity = 0
    clusters = set(clustering.values())
    m = G.number_of_edges()
    for cluster in clusters :
        nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]
        subgraph = G.subgraph(nodes_in_cluster)
        lc = subgraph.number_of_edges()
        d = np.sum([G.degree(node) for node in nodes_in_cluster])
        modularity += (lc/m) - (d/(2*m))**2 
    return modularity



############## Task 8

print("Modularity spectral clustering : ", modularity(gcc, clustering))

random_clustering = dict()
for node in gcc.nodes():
    random_clustering[node] = np.random.randint(0,49)
    
print("Modularity random clustering : ", modularity(gcc, random_clustering))