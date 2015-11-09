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
