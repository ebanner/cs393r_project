import numpy as np

def sigmoid(x):
    """Sigmoid function"""
    
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(f):
    """Sigmoid gradient function
    
    Compute the gradient for the sigmoid function
    
    - f is the sigmoid function value of your original input x
    
    """
    return f * (1-f)

def sigmoid_inverse(z):
    """Computes the inverse of sigmoid

    z is the value that sigmoid produced

    """

    return -np.log(1/z - 1)
