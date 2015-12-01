# Data for training rnn
#
# The idea here is that you should fire on a 2, but only if it's preceded by a
# 1. Context matters!

xs_train = [
        [2,2],
        [1,2]
]

ys_train = [
        [0,0],
        [0,1]
]
