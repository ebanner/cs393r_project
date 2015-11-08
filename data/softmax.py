import numpy as np

m = 2

# positives = [np.random.randn()+1 for _ in range(m//2)]
# negatives = [np.random.randn()-1 for _ in range(m//2)]

positives = [ 1]
negatives = [-1]

xs_train = positives + negatives
ys_train = [1 for _ in range(m//2)] + [0 for _ in range(m//2)]
