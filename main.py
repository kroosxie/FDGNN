import numpy as np

n = 6  # BS num
k = 7  # UE per cell

M = n * k
n_rows = np.floor(n/3)
assign = np.kron(np.arange(n), np.ones((1, k)))

