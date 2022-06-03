import numpy as np
from numpy.linalg import norm, eigvals



x = np.array([1 ,2])
y = np.array([1 ,2])
func = lambda i: np.exp((norm(i)**2))
g = np.zeros((2, 2))
# print(x[1]- y[1])
for k in range(2):
    for j in range(2):
        g[k][j] = func(x[k] - y[j])
print(g)
eigens = eigvals(g)
print(eigens)
