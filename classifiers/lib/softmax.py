import numpy as np

from collections import namedtuple

import logging
from logging import warning as warn

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

Model = namedtuple('Model', ['X', 'ys', 'W', 'b', 'scores', 'probs', 'dscores', 'db', 'dW', 'loss'])
State = namedtuple('State', ['loss', 'dws', 'dbs'])


def softmax_2d(scores):
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

class Softmax:
    """Initialize model parameters
    
    Additionally calculate batch index so we can use minibatches with each training iteration
    
    If you want to inspect the scores after each training example, the pass inspect. If you do
    this then you better set a batch_size to 1. Otherwise you'll only ever get the scores of
    the last training example in the minibatch
    
    """
    def __init__(self, X, ys_train, C, W=None, b=None,
                 learning_rate=0.001, regularizer=1., batch_size=None,
                gradient_checking=False, inspect=False):
        """Initializes softmax classifier
        
        Parameters
        ----------
        X : N x M 2d array containing training input examples
        ys_train : length M list of labels
        C : number of target classes
        W : C x M 2d array of class weights
        b : C length list of biases
        learning_rate : learning rate constant
        regularizer : regularization constant
        batch_size : size of minibatch
        gradient_checking : boolean whether to perform gradient checking during training
        inspect : boolean whether to log all data after every learning session from a training example
        
        """
        (self.N, self.M) = X.shape
        
        self.X_train, self.ys_train = X, ys_train
        self.W = np.random.randn(C, self.N) if not type(W) == np.ndarray else W
        self.b = np.random.randn(C).reshape(C, 1) if not type(b) == np.ndarray else b
        
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        
        self.batch_size = self.M if not batch_size else batch_size
        self.batch_index = 0
        
        self.gradient_checking = gradient_checking
        self.inspect = inspect
        
        # Info from the *last* minibatch that was used to learn from
        self.X, self.ys = None, None
        self.scores, self.dscores = None, None
        self.probs = None
        self.dW, self.dbs = None, None
        self.loss = None
      
    def predict(self, X):
        """Return the prediction of input examples X"""
        
        # Forward Pass (predictions)
        scores = self.W @ X + self.b
        probs = softmax_vectorized(scores)
        
        return probs.argmax(axis=0)
    
    def forward_backward_prop(self, W=None, b=None):
        """Perform forward and backward prop over a minibatch of training examples
        
        Returns loss and gradients
        
        """
        W = self.W if not type(W) == np.ndarray else W
        b = self.b if not type(b) == np.ndarray else b
        
        # Get minibatch of training examples
        low, high = self.batch_index*self.batch_size, (self.batch_index+1)*self.batch_size
        X = self.X_train[:, low:high].reshape(self.N, self.batch_size)
        ys = self.ys_train[low:high]
        
        # Forward Pass (predictions)
        scores = W @ X + b
        probs = softmax_vectorized(scores)
        y_hats = probs[ys, range(self.batch_size)]

        # Loss
        losses = -np.log(y_hats)
        loss = losses.sum()
        loss += (self.regularizer * (0.5*(W**2).sum() + 0.5*(b**2).sum()))

        # Backpropagate to find dw and db
        dscores = probs
        dscores[ys, range(self.batch_size)] -= 1
        
        db = dscores.sum(axis=1, keepdims=True)
        dW = dscores @ X.T
        
        # Regularization
        db += (self.regularizer*b)
        dW += (self.regularizer*W)
        
        # Log additional info?
        if self.inspect:
            self.X, self.ys = X, ys
            self.scores, self.dscores = scores, dscores
            self.dscores[ys, range(self.batch_size)] += 1
            self.probs = probs
        
        return State(loss/self.M, dW/self.M, db/self.M)
    
    def learn(self):
        """Learn from a minibatch of training examples
        
        Run gradient descent on these examples
        
        """        
        loss, dW, db = self.forward_backward_prop()

        self.gradient_check(dW, db)
        
        self.W = self.W - self.learning_rate*dW
        self.b = self.b - self.learning_rate*db
        
        # Update batch index so the next time the next batch in line is used
        self.batch_index = (self.batch_index+1) % (self.M//self.batch_size)
        
        # Log additional info?
        if self.inspect:
            self.dW = dW
            self.db = db
            self.loss = loss
    
    def gradient_check(self, analytic_dW, analytic_db):
        """Verify gradient correctness
        
        The analytic_dws and analytic_dbs come from doing forward-backward
        prop just a second ago. We numerically estimate these gradients on
        the *same* minibatch the analytic gradients were computed from and
        compare them to see if they are close.
        
        Note the same minibatch is being used because this function gets
        called *before* the update to batch_index
        
        """
        if not self.gradient_checking:
            return
        
        numerical_dW, numerical_db = self.numerical_gradients()

        # Compute relative error
        dW_error = abs(numerical_dW - analytic_dW) / (abs(numerical_dW) + abs(analytic_dW))
        db_error = abs(numerical_db - analytic_db) / (abs(numerical_db) + abs(analytic_db))

        try:
            assert(np.linalg.norm(dW_error) < 1e-5 and np.linalg.norm(db_error) < 1e-5)
        except AssertionError:
            warn('Gradient check failed!')
            warn('dW relative error: {}'.format(dW_error))
            warn('db relative error: {}'.format(db_error))
            
    def numerical_gradients(self):
        """Compute numerical gradients of f with respect to self.W and self.b

        Returns approximation for df/dW and df/db

        """
        dW, db = np.zeros_like(self.W), np.zeros_like(self.b)
        h = np.zeros_like(self.W)
        step = 1e-5
        W, b = self.W, self.b
    
        # df/dW
        it = np.nditer(W, flags=['multi_index'])
        while not it.finished:
            ix = it.multi_index
            h[ix] = step
            
            dW[ix] = (self.forward_backward_prop(W+h, b).loss - self.forward_backward_prop(W-h, b).loss) / (2*step)

            h[ix] = 0
            it.iternext()
            
        # df/db
        h = np.zeros_like(self.b)
        it = np.nditer(b, flags=['multi_index'])
        while not it.finished:
            ix = it.multi_index
            h[ix] = step
            
            db[ix] = (self.forward_backward_prop(W, b+h).loss - self.forward_backward_prop(W, b-h).loss) / (2*step)

            h[ix] = 0
            it.iternext()

        return dW, db

    @property
    def info(self):
        """Get a snapshot of the model's most recent activity"""
        
        return Model(self.X, self.ys,
                     self.W, self.b,
                     self.scores, self.probs, self.dscores,
                     self.db, self.dW, self.loss)
