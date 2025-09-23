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

def direction_vector(pj,pi):
    zij = (pj-pi).reshape(-1)
    dij = la.norm(zij)
    gij = zij/dij
    return gij, dij

def multi_vector(N,p,norm=False):
    x = []
    for i in range(len(N)):
        for j in N[i]:
            gij, dij = direction_vector(p[j],p[i])
            x.append(gij if norm==False else dij)
    return np.array(x)

## bearing control
def orth_projection(gij):
    P_gij = np.eye(len(gij)) - np.outer(gij,gij)
    return P_gij

def bearing_rigidity(N,p,dist=1):
    H = incidence_matrix(N)
    P = []
    for i in range(len(N)):
        for j in N[i]:
            gij, dij = direction_vector(p[j],p[i])
            P.append(orth_projection(gij)/dij) if dist==1 \
                else P.append(orth_projection(gij))
    P = block_diag(*P)
    H_bar = np.kron(H,np.eye(len(p[0])))
    Rz = np.dot(P,H_bar)
    return Rz

def MAS_bearing(p,t,N,p_des,dim):
    p = p.reshape(-1,dim)

    # Current formation
    Rz = bearing_rigidity(N,p,dist=0)

    # Desired Bearings
    g_star = multi_vector(N,p_des).reshape(-1)

    # Agent dynamics
    dpdt = np.dot(Rz.T,g_star)

    return dpdt.reshape(-1)

def distance_rigidity(N,p):
    H = incidence_matrix(N)
    z = multi_vector(N,p)
    H_bar = np.kron(H,np.eye(len(p[0])))
    Z = block_diag(*z.reshape(-1,len(z[0])))
    Rp = Z @ H_bar
    return Rp
#*[zij.reshape(1, -1) for zij in z]
# distance control
def MAS_distance(p,t,N,p_des,dim):
    p = p.reshape(-1,dim)

    # Rigidity Matrix
    Rp = distance_rigidity(N,p)

    # Desired distances
    z = multi_vector(N,p,norm=True)
    z_des = multi_vector(N,p_des,norm=True)
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
t = np.arange(0, 6, 0.001)

dim = len(p0[0])
pd = odeint(MAS_distance, p0.reshape(-1), t, args=(N, des_p, dim))
pd = pd.reshape(len(t), len(N), dim)

pb = odeint(MAS_bearing, p0.reshape(-1), t, args=(N, des_p, dim))
pb = pb.reshape(len(t), len(N), dim)


# Recompute edge errors over time
errors = []
for k in range(len(t)):
    z = multi_vector(N, pd[k], norm=True)
    z_des = multi_vector(N, des_p, norm=True)
    e_tild = z**2 - z_des**2
    errors.append(e_tild)
errors = np.array(errors)   # shape (time, edges)

# ------------------------------
# Plotting: Distance vs Bearing
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# --- Distance control ---
# Agent trajectories
for i in range(len(p0)):
    axs[0,0].plot(pd[:, i, 0], pd[:, i, 1])
    axs[0,0].scatter(pd[0, i, 0], pd[0, i, 1], marker='x', c=axs[0,0].lines[-1].get_color())
    axs[0,0].scatter(pd[-1, i, 0], pd[-1, i, 1], marker='o', c=axs[0,0].lines[-1].get_color())
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('y')
axs[0,0].grid()
axs[0,0].set_title("Distance Control - Agent Trajectories")
initial_pt = plt.Line2D([], [], color="black", marker="x", linewidth=0, label="Initial position")
final_pt = plt.Line2D([], [], color="black", marker="o", linewidth=0, label="Final position")
axs[0,0].legend(handles=[initial_pt, final_pt])

# Error evolution (per edge)
for j in range(errors.shape[1]):
    axs[0,1].plot(t, errors[:, j], label=f"Edge {j+1}")
axs[0,1].set_xlabel("Time [s]")
axs[0,1].set_ylabel("Error $e_{ij}(m)$")
axs[0,1].grid()
axs[0,1].set_title("Distance Control - Edge Errors")
axs[0,1].legend()

# --- Bearing control ---
# Agent trajectories
for i in range(len(p0)):
    axs[1,0].plot(pb[:, i, 0], pb[:, i, 1])
    axs[1,0].scatter(pb[0, i, 0], pb[0, i, 1], marker='x', c=axs[1,0].lines[-1].get_color())
    axs[1,0].scatter(pb[-1, i, 0], pb[-1, i, 1], marker='o', c=axs[1,0].lines[-1].get_color())
axs[1,0].set_xlabel('x')
axs[1,0].set_ylabel('y')
axs[1,0].grid()
axs[1,0].set_title("Bearing Control - Agent Trajectories")
axs[1,0].legend(handles=[initial_pt, final_pt])

# Bearing has no scalar "distance error", so we can instead check
# deviation from desired bearings (angle error per edge)
bearing_errors = []
for k in range(len(t)):
    g = multi_vector(N, pb[k])        # current unit bearings
    g_des = multi_vector(N, des_p)    # desired bearings
    err = [1 - np.dot(g[i], g_des[i]) for i in range(len(g))]  # angle mismatch measure
    bearing_errors.append(err)
bearing_errors = np.array(bearing_errors)

for j in range(bearing_errors.shape[1]):
    axs[1,1].plot(t, bearing_errors[:, j], label=f"Edge {j+1}")
axs[1,1].set_xlabel("Time [s]")
axs[1,1].set_ylabel("Bearing Error")
axs[1,1].grid()
axs[1,1].set_title("Bearing Control - Edge Errors")
axs[1,1].legend()

plt.tight_layout()
plt.show()

# ------------------------------
# --- Solution check ---
# pd, pb are (T, n, 2) arrays final states
p_dist_final = pd[-1]   # (n,2)
p_bearing_final = pb[-1]

# 1) Pairwise distances
def pairwise_distances(P):
    n = P.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(P[i]-P[j])
    return D

D_des = pairwise_distances(des_p)
D_pd  = pairwise_distances(p_dist_final)
D_pb  = pairwise_distances(p_bearing_final)

# look at norms of differences
print("distance ctrl distance error norm:", np.linalg.norm(D_pd - D_des))
print("bearing ctrl distance error norm: ", np.linalg.norm(D_pb - D_des))

# 2) Pairwise bearings (unit vectors)
def pairwise_bearings(P):
    n = P.shape[0]
    G = []
    for i in range(n):
        for j in range(n):
            if i!=j:
                v = P[j]-P[i]
                G.append(v/np.linalg.norm(v))
    return np.array(G)

G_des = pairwise_bearings(des_p)
G_pd  = pairwise_bearings(p_dist_final)
G_pb  = pairwise_bearings(p_bearing_final)

print("distance ctrl bearing mismatch:", np.linalg.norm(G_pd - G_des))
print("bearing ctrl bearing mismatch: ", np.linalg.norm(G_pb - G_des))

def align_similarity(X, Y):
    # Find s,R,t to best map Y -> X in least-squares sense (Procrustes with scaling)
    # X, Y: (n,2) corresponding points
    n = X.shape[0]
    # centroids
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    # compute scale & rotation via SVD of Yc^T Xc
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = U @ Vt
    # scale
    varY = (Yc**2).sum()
    s = S.sum() / varY
    t = muX - s*(R @ muY)
    Y_aligned = s*(Y @ R.T) + t
    return Y_aligned, s, R, t

# Example: align bearing final to desired
Y_aligned, s, R, t = align_similarity(des_p, p_bearing_final)
print("bearing final aligned to desired, scale:", s)
print("RMS error after alignment:", np.sqrt(np.mean(np.sum((Y_aligned - des_p)**2, axis=1))))
