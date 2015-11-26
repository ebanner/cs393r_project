import numpy as np

from softmax import softmax_vectorized

from nn.helper import State, Model, sigmoid, random_Ws, random_bs
from nn.units import Affine, Sigmoid, SoftmaxCrossEntropy

import logging
from logging import warning as warn

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class NeuralNetwork:
    """Initialize model parameters
    
    Additionally calculate batch index so we can use minibatches with each training iteration
    
    If you want to inspect the scores after each training example, the pass inspect. If you do
    this then you better set a batch_size to 1. Otherwise you'll only ever get the scores of
    the last training example in the minibatch
    
    """
    def __init__(self, X, ys_train, H, C,
            params=None, learning_rate=0.001, regularizer=1., batch_size=None,
            gradient_checking=False, inspect=False):
        """Initializes neural network classifier
        
        Parameters
        ----------
        X : N x M 2d array containing training input examples
        ys_train : length M list of labels
        H : size of hidden layer
        L : number of hidden layers
        C : number of target classes

        params : dict of params that capture the neural network. dict should
        look like this:
            {'W': [W_1, W_2, ..., W_L],
             'b': [b1, b2, ..., b_L]}
        Let h_i and h_j be the sizes of the layers of hidden layer i and j where
        j = i+1. Then W_i has shape h_j x N, where N is the minibatch size.

        learning_rate : learning rate constant
        regularizer : regularization constant
        batch_size : size of minibatch
        gradient_checking : boolean whether to perform gradient checking during training
        inspect : boolean whether to log all data after every learning session from a training example
        
        """
        (self.N, self.M) = X.shape
        self.C = C
        
        self.X_train, self.ys_train = X, ys_train

        self.batch_size = self.M if not batch_size else batch_size
        self.batch_index = 0

        # Initialize params?
        layer_sizes = [self.N] + H + [self.C]
        self.params = {
                'Ws': np.array(list(random_Ws(layer_sizes))),
                'bs': np.array(list(random_bs(layer_sizes))),
        }

        self.learning_rate = learning_rate
        self.regularizer = regularizer
        
        self.gradient_checking = gradient_checking
        self.inspect = inspect

        # Initialize the friggin' network
        L = len(H)
        self.affines = [Affine() for _ in range(1+L)]
        self.sigmoids = [Sigmoid() for _ in range(L)]
        self.softmax_ce = SoftmaxCrossEntropy()
        
        # Info from the *last* minibatch that was used to learn from
        self.X, self.ys = None, None
        self.Z, self.hidden = None, None
        self.scores, self.dscores = None, None
        self.probs = None
        self.loss = None
        
    def predict(self, X):
        """Return the probability of x belonging to either class"""
        
        hidden = sigmoid(self.Wh @ X + self.bh)
        scores = self.Ws @ hidden + self.bs
        probs = softmax_vectorized(scores)
        
        return probs.argmax(axis=0)
        
    def forward_backward_prop(self, params=None):
        """Perform forward and backward prop over a minibatch of training examples
        
        Returns loss and gradients
        
        """
        params = self.params if not params else params
        Ws, bs = params['Ws'], params['bs']
        affines, sigmoids, softmax_ce = self.affines, self.sigmoids, self.softmax_ce

        # Get minibatch of training examples
        low, high = self.batch_index*self.batch_size, (self.batch_index+1)*self.batch_size
        X = self.X_train[:, low:high].reshape(self.N, self.batch_size)
        ys = self.ys_train[low:high]

        # Forward pass
        hidden = X
        for W, b, affine, sigmoid in zip(Ws, bs, affines, sigmoids):
            Z = affine.forward(hidden, W, b)
            hidden = sigmoid.forward(Z)

        # Power through to the doorstep of the softmax layer
        scores = affines[-1].forward(hidden, Ws[-1], bs[-1])

        # Softmax + Cross Entropy layer
        loss, losses, probs = softmax_ce.forward(scores, ys)
        
        # Backprop!
        dscores = softmax_ce.backward(1)
        dhidden = affines[-1].backward(dscores)
        for affine, sigmoid in zip(reversed(affines[:-1]), reversed(sigmoids)):
            dZ = sigmoid.backward(dhidden)
            dhidden = affine.backward(dZ)

        # Accumulate gradients
        dWs = np.array([affine.dW for affine in affines])
        dbs = np.array([affine.db for affine in affines])

        # Regularization
        loss += self.regularizer * \
                0.5*(np.sum(np.sum(dW**2) for dW in dWs) + np.sum(np.sum(db**2) for db in dbs))
        dWs += self.regularizer*Ws
        dbs += self.regularizer*bs

        gradients = {'loss': loss/self.M, 'dWs': dWs/self.M, 'dbs': dbs/self.M}
        
        # Log additional info?
        if self.inspect:
            self.X, self.ys = X, ys
            self.gradients = gradients

        return gradients
    
    def learn(self):
        """Learn from a minibatch of training examples
        
        Run gradient descent on these examples
        
        """
        gradients = self.forward_backward_prop()

        self.gradient_check(gradients)

        # Step
        for (p_symbol, param), (g_symbol, _) in zip(sorted(self.params.items()), sorted(gradients.items())):
            for i, p in enumerate(param):
                self.params[p_symbol][i] = p - self.learning_rate*gradients[g_symbol][i]

        # Update batch index so the next time the next batch in line is used
        self.batch_index = (self.batch_index+1) % (self.M//self.batch_size)
        
        # Log additional info?
        if self.inspect:
            self.gradient = gradients
    
    def gradient_check(self, analytic_gradients):
        """Verify gradient correctness
        
        The analytic gradients come from doing forward-backward prop just a
        second ago. We numerically estimate these gradients on the *same*
        minibatch the analytic gradients were computed from and compare them to
        see if they are close.
        
        Note the same minibatch is being used because this function gets
        called *before* the update to batch_index
        
        """
        if not self.gradient_checking:
            return
        
        # Compute analytic gradients for all parameters
        numerical_gradients = {p_symbol: self.numerical_gradients(p_symbol) for p_symbol in ('Ws', 'bs')}

        # Compute relative error
        for (p_symbol, num_grad), (p_symbol, ana_grad) in \
        zip(sorted(analytic_gradients.items()), sorted(numerical_gradients.items())):
            errors = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad))
            if np.sum(np.linalg.norm(error) for error in errors) > 1e-5:
                warn('Gradient check failed!')
                warn('{} relative error: {}'.format(p_symbol, errors))
            
    def numerical_gradients(self, param_symbol):
        """Compute numerical gradients of L with respect to param_symbol

        Returns approximation for dL/dparam

        For example, param_symbol may be 'Ws'. Then for every W in Ws, we'll
        compute dL/dW. That amounts to computing dL/dw for every little w in W

        """
        param = self.params[param_symbol]
        dps = param[:]
        params = {'Ws': self.params['Ws'], 'bs': self.params['bs']}
        
        step = 1e-5
    
        # dL/dparam
        for i, p in enumerate(param):
            h = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'])
            while not it.finished:
                ix = it.multi_index
                h[ix] = step

                # L(p+h)
                params[param_symbol][i] += h
                loss_forward = self.forward_backward_prop(params)['loss']

                # L(p-h)
                params[param_symbol][i] -= 2*h
                loss_backward = self.forward_backward_prop(params)['loss']

                # dL/dp = L(p+h)-L(p-h) / 2*h
                dps[i][ix] = (loss_forward-loss_backward) / (2*step)

                # Cleanup
                params[param_symbol][i] += h
                h[ix] = 0

                it.iternext()
            
        return dps

    @property
    def info(self):
        """Get a snapshot of the model's most recent activity"""
        
        return Model(self.X, self.ys,
                self.parmams, self.gradients, self.loss)
