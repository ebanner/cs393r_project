import numpy as np
from collections import namedtuple

# Model = namedtuple('Model',['x', 'y', 'wh', 'bh', 'z', 'w1', 'b1', 'score1', 'w2', 'b2', 'score2', 'prob1', 'prob2', 'dscore1', 'dscore2', 'db1', 'dw1', 'db2', 'dw2', 'loss'])
# State = namedtuple('State', ['loss', 'dwh', 'dbh', 'dws', 'dbs'])

Model = namedtuple('Model', ['X', 'ys',
                             'Wh', 'bh', 'Z', 'hidden',
                             'Ws', 'bs',
                             'scores', 'probs', 'dscores',
                             'dbs', 'dWs',
                             'dhidden', 'dZ',
                             'dbh', 'dWh',
                             'loss'])
State = namedtuple('State', ['loss', 'dWh', 'dbh', 'dWs', 'dbs'])

def sigmoid(x):
    """Sigmoid function"""
    
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(f):
    """Sigmoid gradient function
    
    Compute the gradient for the sigmoid function
    
    - f is the sigmoid function value of your original input x
    
    """
    return f * (1-f)

def tanh_grad(f):
    """tanh gradient function
    
    Compute the gradient for the sigmoid function
    
    - f is the sigmoid function value of your original input x
    
    """
    return 1 - f**2


def sigmoid_inverse(z):
    """Computes the inverse of sigmoid

    z is the value that sigmoid produced

    """

    return -np.log(1/z - 1)
