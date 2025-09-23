'''
This file is made in conjunction with 
Control of Multi-agent Systems - Theory and Simulations with Python

Chapter - 3: Consensus Control
Hand notes will be taken on paper while worked python examples given in the text will be worked here 
'''
# # Continuous Time Consensus
# from scipy. integrate import odeint
# import numpy as np
# import matplotlib. pyplot as plt

# def MAS(x,t,N):
#     dxdt = [0] * len(N)
#     u = [0] * len(N)

#     # Definition of agent i
#     for i in range(len(N)):
#         # Computation of the control input of agent i
#         dif = []
#         for j in N[i]:
#             dif.append(x[j] - x[i])
#             u[i] = sum(dif)

#             # Dynamics of agent i
#             dxdt[i] = u[i]

#     return dxdt

# N = [[2], [3,5], [4], [1,2], [6], [2]]
# x0 = [-1, 2, 6, 3, -3, 1]
# t = np.arange(0, 5, 0.001)
# N.insert(0 ,[])
# x0.insert(0,0)
# x = odeint(MAS, x0, t, args=(N,))

# plt.plot(t,np.delete(x, 0, 1))
# plt.xlabel('t')
# plt.ylabel('xi')
# plt.grid()
# plt.show() 

# # Discrete Time Consensus
# from scipy. integrate import odeint
# import numpy as np
# import matplotlib. pyplot as plt

# def MAS(x,N,e):
#     x_next = [0] * len(N)
#     u = [0] * len(N)

#     # Definition of agent i
#     for i in range(len(N)):
#         # Computation of the control input of agent i
#         dif = []

#         for j in N[i]:
#             dif.append(x[j] - x[i])

#             u[i] = e * sum(dif)

#             # Dynamics of agent i
#             x_next[i] = x[i] + u[i]

#     return x_next

# N = [[2], [3,5], [4], [1,2], [6], [2]]
# e = 0.4
# x0 = [-1, 2, 6, 3, -3, 1]
# K = np.arange(0, 11, 1)

# N.insert(0 ,[])
# x0.insert(0,0)
# X = []
# xk = x0
# for k in K:
#     x_next = MAS(xk ,N,e)
#     X.append(xk)
#     xk = x_next

# plt.plot(K,np.delete(X, 0, 1))
# plt.xlabel('k')
# plt.ylabel('xi')
# plt.grid()
# plt.show() 

# # Continuous Time Alternate
# from scipy. integrate import odeint
# import numpy as np
# import matplotlib. pyplot as plt

# def MAS(x,t,L):
#     dxdt = np.dot(-L,x)
#     return dxdt

# L = np.array([[ 1, -1, 0, 0, 0, 0],
#               [ 0, 2, -1, 0, -1, 0],
#               [ 0, 0, 1, -1, 0, 0],
#               [-1, -1, 0, 2, 0, 0],
#               [ 0, 0, 0, 0, 1, -1],
#               [ 0, -1, 0, 0, 0, 1]])

# x0 = np.array([-1, 2, 6, 3, -3, 1])
# t = np.arange(0, 5, 0.001)
# x = odeint(MAS , x0 , t, args=(L,))

# print(sum(x0)/len(x0))

# plt.plot(t,x)
# plt.xlabel('t')
# plt.ylabel('xi')
# plt.grid()
# plt.show()

# Continuous Time Alternate
from scipy.integrate import odeint
import numpy as np
import numpy.linalg as LA
import matplotlib. pyplot as plt

def MAS(x,t,L):
    dxdt = np.dot(-L,x)
    return dxdt

L = np.array([[ 1, -1, 0, 0, 0, 0],
              [ 0, 2, -1, 0, -1, 0],
              [ 0, 0, 1, -1, 0, 0],
              [-1, -1, 0, 2, 0, 0],
              [ 0, 0, 0, 0, 1, -1],
              [ 0, -1, 0, 0, 0, 1]])

x0 = np.array([-1, 2, 6, 3, -3, 1])
t = np.arange(0, 5, 0.001)
x = odeint(MAS , x0 , t, args=(L,))

s, V = LA.eig(L.T)
print(V[:,0]/min(abs(V[:,0])))
v1 = V[:,0]
print(np.dot(v1,x0)/sum(v1))

plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('xi')
plt.grid()
plt.show()