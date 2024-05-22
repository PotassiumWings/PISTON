import numpy as np

r = 15
c = 5
adj = [[0 for i in range(r * c)] for j in range(r * c)]
for i in range(r):
    for j in range(c):
        if i < r - 1:
            adj[i * c + j][(i + 1) * c + j] = adj[(i + 1) * c + j][i * c + j] = 1
        if j < c - 1:
            adj[i * c + j][i * c + j + 1] = adj[i * c + j + 1][i * c + j] = 1

np.savez_compressed("../data/NYC-TOD4/adj_mx.npz", adj_mx=adj)
