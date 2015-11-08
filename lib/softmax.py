import numpy as np

def softmax(scores):
    """Compute the softmax between two numbers
    
    s1 is the number we're finding the softmax of
    
    """
    e_x = np.exp(scores - scores.max())
    
    return e_x / e_x.sum()
