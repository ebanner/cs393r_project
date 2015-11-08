import numpy as np

m = 50

xs_train = np.linspace(5, 30, num=m)
w_true, b = np.array([-2.0]), 50

# Compute ys
noise_level = 10
ys_train = w_true[0]*xs_train + b
ys_train = np.array([y + np.random.randn()*noise_level for y in ys_train])
