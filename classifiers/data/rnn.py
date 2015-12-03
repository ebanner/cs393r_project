import numpy as np

# Data for training rnn
#
# The idea here is that you should fire on a 2, but only if it's preceded by a
# 1. Context matters!

xs_train = [
    ( 1, 1),
    (-1, 1)
]

ys_train = [
    (0, 0),
    (1, 1)
]

# The goal here is to classify 1 always as 0, except when it's followed by a -1.
# The class 2 is introduced as a special marker class
X_train = np.array([
    [1, 1, -1, 1],
])

Y_train = np.array(
    [0, 0,  2, 1]
)

long_X_train = np.array([np.random.randint(10) for _ in range(1000)])
long_Y_train = np.array([np.random.randint(2) for _ in range(1000)])
