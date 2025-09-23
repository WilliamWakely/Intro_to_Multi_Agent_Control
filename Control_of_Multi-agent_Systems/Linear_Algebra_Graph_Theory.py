'''
This file is made in conjunction with 
Control of Multi-agent Systems - Theory and Simulations with Python

Chapter - 2: Review of Linear Algebra and Graph Theory
Hand notes will be taken on paper while worked python examples given in the text will be worked here 
'''

import numpy as np
import numpy.linalg as LA
import networkx as nx
# from networkx.drawing.nx_pydot import graphviz_layout

# Eigen value/vector review

# # Example 1
# A = np.array([[1,0,0],[0,1,1],[0,0,1]])
# s, V = LA.eig(A)
# print(s)
# print(V)

# # Jordan Canonical Form and Spectral Decomposition
# from sympy import Matrix
# A = np.array([[-1,-7,2],[1,4,-1],[1,2,0]])
# M = Matrix(A)
# T, J = M.jordan_form()
# print(T)
# print(J)

# # Graph Laplacian
# G = nx.DiGraph()
# V = [1,2,3,4]
# E = [(1,2),(1,3),(3,2),(3,4),(4,1)]
# G.add_nodes_from(V)
# G.add_edges_from(E)

# print(nx.adjacency_matrix(G))
# print(nx.adjacency_matrix(G).todense())
# print(nx.laplacian_matrix(G).todense())