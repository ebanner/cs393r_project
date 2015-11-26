import numpy as np

from softmax import softmax_2d
from nn.shallow.helper import sigmoid, sigmoid_grad

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
    
    def forward(self, a, w, b):
        self.a, self.w, self.b = a, w, b
        
        return w*a + b
    
    def backward(self, error):
        db = error
        dw = self.a * error
        da = self.w * error
        
        return da, dw, db
    
class Sigmoid(Unit):
    """Sigmoid non-linearity"""
    
    def forward(self, z):
        self.h = sigmoid(z)
        
        return self.h
    
    def backward(self, error):
        dz = sigmoid_grad(self.h) * error
        
        return dz
    
class Softmax(Unit):
    """Converts input into a probability distribution"""
    
    def forward(self, scores):
        self.prob1, self.prob2 = softmax_2d(np.array(scores))
        
        return self.prob1, self.prob2
    
    def backward(self, dprobs):
        dscore1 = sigmoid_grad(self.prob1)*dprobs[0] + (-self.prob1*self.prob2)*dprobs[1]
        dscore2 = (-self.prob1*self.prob2)*dprobs[0] + sigmoid_grad(self.prob2)*dprobs[1]
        
        return dscore1, dscore2
    
class CrossEntropy(Unit):
    """Cross Entropy loss"""
    
    def forward(self, probs, y):
        """Forward pass (loss)
        
        Parameters
        ----------
        probs : tuple containing the probability for each class
        y : the index of the correct class
        
        """
        self.probs = probs
        self.y = y
        
        loss = -np.log(probs[y])
        
        return loss
    
    def backward(self, dloss):
        """Compute backward pass
        
        The only prob that can affect the loss is the probability for the correct class
        
        """
        dprobs = np.array(self.probs)
        dprobs[self.y] = -1/self.probs[self.y]
        dprobs[(self.y+1)%2] = 0
        
        return dprobs * dloss
