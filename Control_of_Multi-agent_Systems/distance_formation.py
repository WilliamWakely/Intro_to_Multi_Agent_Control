'''
This file is made in conjunction with 
Control of Multi-agent Systems - Theory and Simulations with Python

Chapter - 5: Formation Control
Hand notes will be taken on paper while worked python examples given in the text will be worked here 
'''

import numpy as np
from numpy import linalg as la 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import block_diag

# Functions 
def incidence_matrix(N):
    N_flat = [x for y in N for x in y]
    H = np.zeros([len(N_flat),len(N)])
    
    edge_count = 0
    for i in range(len(N)):
        for j in N[i]:
            H[edge_count,i] = -1
            H[edge_count,j] = 1
            edge_count += 1
    return H

def distance_vector(pj,pi):
    zij = (pj-pi).reshape(-1)
    return zij

def multi_distance_vector(N,p,norm=False):
    z = []
    for i in range(len(N)):
        for j in N[i]:
            zij = distance_vector(p[j],p[i])
            z.append(zij) if norm==False else z.append(la.norm(zij))
    return np.array(z)

def distance_rigidity(N,p):
    H = incidence_matrix(N)
    z = multi_distance_vector(N,p)
    H_bar = np.kron(H,np.eye(len(p[0])))
    Z = block_diag(*[zij.reshape(1, -1) for zij in z])
    Rp = Z @ H_bar
    return Rp

# distance control
def MAS_distance(p,t,N,p_des,dim):
    p = p.reshape(-1,dim)

    # Rigidity Matrix
    Rp = distance_rigidity(N,p)

    # Desired distances
    z = multi_distance_vector(N,p,norm=True)
    z_des = multi_distance_vector(N,p_des,norm=True)
    e_tild = z**2 - z_des**2

    # Agent dynamics
    dpdt = -np.dot(Rp.T,e_tild)

    return dpdt.reshape(-1)

# ------------------------------
# Connectivity
N = [[2,4], [1,3,4,5], [2,5], [1,2,5], [2,3,4]] 
N = [[x-1 for x in y] for y in N]

# Simulation Setup
p0 = np.array ([[0,2], [0,0], [1,-1], [1,1], [3,0]])
des_p = np.array ([[0,1], [0,0], [1,-1], [1,1], [2,0]])
t = np.arange(0, 1, 0.001)

dim = len(p0[0])
p = odeint(MAS_distance, p0.reshape(-1), t, args=(N, des_p, dim))
p = p.reshape(len(t), len(N), dim)

# Recompute edge errors over time
errors = []
for k in range(len(t)):
    z = multi_distance_vector(N, p[k], norm=True)
    z_des = multi_distance_vector(N, des_p, norm=True)
    e_tild = z**2 - z_des**2
    errors.append(e_tild)
errors = np.array(errors)   # shape (time, edges)

# ------------------------------
# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Agent trajectories
for i in range(len(p0)):
    axs[0].plot(p[:, i, 0], p[:, i, 1])
    axs[0].scatter(p[0, i, 0], p[0, i, 1], marker='x', c=axs[0].lines[-1].get_color())
    axs[0].scatter(p[-1, i, 0], p[-1, i, 1], marker='o', c=axs[0].lines[-1].get_color())
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
initial_pt = plt.Line2D([], [], color="black", marker="x", linewidth=0, label="Initial position")
final_pt = plt.Line2D([], [], color="black", marker="o", linewidth=0, label="Final position")
axs[0].legend(handles=[initial_pt, final_pt])
axs[0].grid()
axs[0].set_title("Agent Trajectories")

# Error evolution (per edge)
for j in range(errors.shape[1]):
    axs[1].plot(t, errors[:, j], label=f"Edge {j+1}")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Error $e_{ij}(m)$")
axs[1].grid()
axs[1].set_title("Edge Errors Over Time")
axs[1].legend()

plt.tight_layout()
plt.show()