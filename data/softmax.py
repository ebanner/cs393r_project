import numpy as np

m = 2

## 1D
positives = [ 1]
negatives = [-1]

xs_train = positives + negatives
ys_train = [1 for _ in range(m//2)] + [0 for _ in range(m//2)]

## Vectorized

positives = [[1,1], [2,2]]
negatives = [[2,1], [1,2]]

X_train = np.array(positives+negatives).T
Y_train = [0, 0, 1, 1]
