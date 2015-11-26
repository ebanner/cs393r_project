import numpy as np

from collections import namedtuple

from nn.shallow.helper import sigmoid, sigmoid_grad
from softmax import softmax_vectorized


class Unit:
    """Interface for a unit in a neural network
    
    To be a unit in a neural network, you have to implement a forward and backward pass
    
    """
    def forward(self):
        pass
    def backward(self):
        pass
    
class Affine(Unit):
    """Multiplication of weight with activation and addition of a bias term"""
    
    def forward(self, A, W, b):
        self.A, self.W, self.b = A, W, b
        
        return W @ A + b
    
    def backward(self, error):
        db = error.sum(axis=1, keepdims=True)
        dW = error @ self.A.T
        dA = self.W.T @ error
        
        return dA, dW, db
    
class Sigmoid(Unit):
    """Sigmoid non-linearity"""
    
    def forward(self, Z):
        self.h = sigmoid(Z)
        
        return self.h
    
    def backward(self, error):
        dZ = sigmoid_grad(self.h) * error
        
        return dZ
    
class SoftmaxCrossEntropy(Unit):
    """Converts input into a probability distribution"""
    
    def forward(self, scores, ys):
        self.ys = ys
        self.probs = softmax_vectorized(np.array(scores))
        y_hats = self.probs[self.ys, range(len(self.ys))]

        # Loss
        losses = -np.log(y_hats)
        loss = losses.sum()
        
        return loss, losses, self.probs
    
    def backward(self, dloss):
        dscores = self.probs
        dscores[self.ys, range(len(self.ys))] -= 1
        
        return dscores
