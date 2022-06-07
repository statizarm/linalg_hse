import numpy as np
from numpy import linalg as la


class PageRank:
    def __init__(self, b=0.15, e=1e-3, n_iter=int(1e9)):
        self.b = b
        self.e = e
        self.n_iter = n_iter


    def __call__(self, adj):
        assert len(adj.shape) == 2
        assert adj.shape[0] == adj.shape[1]

        K = np.zeros(adj.shape)
        for i in range(adj.shape[0]):
            out_degree = 0
            for j in range(adj.shape[0]):
                if adj[i,j] > 0:
                    out_degree += 1
            K[i, i] = out_degree

        M = (la.inv(K) @ adj).transpose()
        N = adj.shape[0]
        v = np.ones((N,1)) / N
        d = self.b * np.ones((N,1)) / N
        for _ in range(self.n_iter):
            new_v = (1 - self.b) * (M @ v) + d
            if la.norm(new_v - v) < self.e:
                v = new_v
                break
            v = new_v

        return v, M
