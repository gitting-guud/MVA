"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
G=nx.read_edgelist("D:\MVA\Altegrad\Lab_5\code/datasets/CA-HepTh.txt",delimiter = "\t")
print("Number of nodes : ", G.number_of_nodes())
print("-----------------------")
print("Number of edges : ", G.number_of_edges())
print("-----------------------------------------------------------------")
############## Task 2

print("The number of connected components : ", len(list(nx.connected_components(G))))
gcc = max(nx.connected_component_subgraphs(G), key=len)
print("Fraction of nodes of the GCC: ", gcc.number_of_nodes()/G.number_of_nodes())
print("Fration of edges of the GCC: ", gcc.number_of_edges()/G.number_of_edges())
print("-----------------------------------------------------------------")

############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("Min number of degrees in the graph : ", np.min(degree_sequence))
print("Max number of degrees in the graph : ", np.max(degree_sequence))
print("Median number of degrees in the graph : ", np.median(degree_sequence))
print("Average number of degrees in the graph : ", np.mean(degree_sequence))
plt.figure(figsize= (15,7))
plt.grid()
plt.title("Distribution of degrees of the graph")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.plot(nx.degree_histogram(G));
plt.show()

print("-----------------------------------------------------------------")


############## Task 4

plt.figure(figsize = (7,5))
plt.grid()
plt.title("Distribution of degrees of the graph in log-log scale")
plt.xlabel("log-Degree")
plt.ylabel("log-Frequency")
plt.loglog(nx.degree_histogram(G));
##################
# your code here #
##################