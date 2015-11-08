import numpy as np

def softmax(scores):
    """Compute the softmax between two numbers
    
    s1 is the number we're finding the softmax of
    
    """
    e_x = np.exp(scores - scores.max())
    
    return e_x / e_x.sum()

def softmax_vectorized(scores):
    """Softmax function

    Assumes the scores for each training example go vertically

    """
    e_x = np.exp(scores - scores.max(axis=0))

    return e_x / e_x.sum(axis=0)
