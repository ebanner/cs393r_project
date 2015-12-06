import numpy as np

from nn.shallow.helper import tanh_grad
from softmax import softmax_vectorized

from rnn.support import State, Snapshot, Gradients

import logging
from logging import warning as warn

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class RecurrentNeuralNetwork:
    """Initialize model parameters
    
    Additionally calculate batch index so we can use minibatches with each training iteration
    
    If you want to inspect the scores after each training example, the pass inspect. If you do
    this then you better set a batch_size to 1. Otherwise you'll only ever get the scores of
    the last training example in the minibatch
    
    """
    def __init__(self, X, ys_train, H, C,
                 Whh=None, bhh=None, Wxh=None, bxh=None, Ws=None, bs=None,
                 rollout=None, learning_rate=0.001, regularizer=1.,
                gradient_checking=False, inspect=False):
        """Initializes recurrent neural network classifier
        
        Parameters
        ----------
        X : N x M 2d array containing all elements in the sequence
        ys_train : length M list of labels
        H : size of hidden layer
        C : number of target classes
        Whh : H x H 2d matrix mapping previous hidden layer to new hidden layer
        bhh : H x 1 array of bias terms applied after Whh multiplication
        Wxh : H x N 2d matrix mapping input at time t to hidden size
        bxh : H x 1 array of bias terms applied after Wxh multiplication
        Ws : C x H matrix of softmax weights
        bs : C x 1 array of softmax biases
        rollout : the number of tokens to train the rnn on in one go
        learning_rate : learning rate constant
        regularizer : regularization constant
        gradient_checking : boolean whether to perform gradient checking during training
        inspect : boolean whether to log all data after every learning session from a training example
        
        """
        (self.N, self.T) = X.shape
        self.H = H
        
        self.X_train, self.ys_train = X, ys_train

        # Hidden and input weights
        self.Whh = np.identity(H) if not type(Whh) == np.ndarray else Whh
        self.bhh = np.random.uniform(-.8, .8, (H, 1)) if not type(bhh) == np.ndarray else bhh
        self.Wxh = np.random.uniform(-.8, .8, (H, self.N)) if not type(Wxh) == np.ndarray else Wxh
        self.bxh = np.random.uniform(-.8, .8, (H, 1)) if not type(bxh) == np.ndarray else bxh

        # Current hidden state
        self.hidden = np.zeros((H, 1))
        
        # Softmax weights
        self.Ws = np.random.randn(C, H) if not type(Ws) == np.ndarray else Ws
        self.bs = np.random.randn(C, 1) if not type(bs) == np.ndarray else bs
        
        self.rollout = self.T if not rollout else rollout
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        
        self.gradient_checking = gradient_checking
        self.inspect = inspect
        
        self.train_index = 0
        
    def predict(self, X):
        """Return the probability of x belonging to either class"""
        
        # Create artificial labels just to make forward_backward_prop() happy
        N, T = X.shape
        ys = np.ones(T, dtype=np.int)
        
        new_hidden, scores = self.forward_backward_prop(X=X, ys=ys, rollout=T, predict=True)
        proper_scores = np.hstack([score for t, score in sorted(scores.items())])

        # Update the hidden state
        self.hidden = new_hidden
        
        return proper_scores, proper_scores.argmax(axis=0)
        
    def forward_backward_prop(self, X=None, ys=None, rollout=None, train_index=None,
            Whh=None, bhh=None, Wxh=None, bxh=None, Ws=None, bs=None,
            hidden=None, predict=False):
        """Perform forward and backward prop over a single training example
        
        Returns loss and gradients
        
        """
        # Hidden and input weights
        Whh = self.Whh if not type(Whh) == np.ndarray else Whh
        bhh = self.bhh if not type(bhh) == np.ndarray else bhh
        Wxh = self.Wxh if not type(Wxh) == np.ndarray else Wxh
        bxh = self.bxh if not type(bxh) == np.ndarray else bxh
        
        # Softmax weights
        Ws = self.Ws if not type(Ws) == np.ndarray else Ws
        bs = self.bs if not type(bs) == np.ndarray else bs
        
        # Initial hidden state
        hidden = self.hidden if not type(hidden) == np.ndarray else hidden

        # Where to start in the sequence and how far to go
        rollout = self.rollout if not rollout else rollout
        train_index = self.train_index if not train_index else train_index

        # Get next portion of sequence to train on
        if not type(X) == np.ndarray:
            X = self.X_train[:, train_index:train_index+rollout] 
            ys = self.ys_train[train_index:train_index+rollout]
            
            # Got to the end and need to wrap around?
            if train_index+rollout > self.T:
                rollover_index = (train_index+rollout) % self.T

                X = np.hstack([X, self.X_train[:, :rollover_index]])
                ys = np.hstack([ys, self.ys_train[:rollover_index]])

        # Append column of zeros to align X and Y with natural time
        X, ys = np.hstack([np.zeros((self.N, 1)), X]), np.hstack([np.zeros(1, dtype=np.int), ys])

        # Forward pass!
        dWhh, dbhh = np.zeros_like(Whh), np.zeros_like(bhh)
        dWxh, dbxh = np.zeros_like(Wxh), np.zeros_like(bxh)
        dWs, dbs = np.zeros_like(Ws), np.zeros_like(bs)
        
        loss = 0.
        hiddens = {0: hidden}
        dhiddens, dhiddens_downstream, dhiddens_local = {}, {rollout:np.zeros((self.H, 1))}, {}
        scores, probs = {}, {}
        for t in range(1, rollout+1):
            # Previous hidden layer and input at time t
            Z = (Whh @ hiddens[t-1] + bhh) + (Wxh @ X[:,[t]] + bxh)
            hiddens[t] = np.tanh(Z)
            
            # Softmax
            scores[t] = Ws @ hiddens[t] + bs
            probs[t] = softmax_vectorized(scores[t])
            y_hat = probs[t][ys[t]]

            # Loss
            loss += -np.log(y_hat).sum()

        # Add regularization
        loss += self.regularizer * 0.5*(np.sum(Whh**2) + np.sum(bhh**2) +
                                        np.sum(Wxh**2) + np.sum(bxh**2) +
                                        np.sum(Ws**2) + np.sum(bs**2))
        if predict:
            return hiddens[rollout], scores
        
        # Backpropagate!
        backwards = list(reversed(range(rollout+1)))
        for t in backwards[:-1]:
            # Scores
            dscores = probs[t]
            dscores[ys[t], 0] -= 1

            # Softmax weights
            dbs += dscores
            dWs += dscores @ hiddens[t].T

            dhiddens_local[t] = Ws.T @ dscores
            dhiddens[t] = dhiddens_local[t] + dhiddens_downstream[t] # Karpathy optimization
            
            dZ = tanh_grad(hiddens[t]) * dhiddens[t]

            # Input and hidden weights
            dbxh += dZ
            dWxh += dZ @ X[:,[t]].T
            dbhh += dZ
            dWhh += dZ @ hiddens[t-1].T
            
            # Set up incoming hidden weight gradient for previous time step
            dhiddens_downstream[t-1] = Whh.T @ dZ
        
        # Regularization
        #
        # Hidden and input weights
        dWhh += (self.regularizer*Whh)
        dbhh += (self.regularizer*bhh)
        dWxh += (self.regularizer*Wxh)
        dbxh += (self.regularizer*bxh)
        
        # Softmax weights
        dWs += (self.regularizer*Ws)
        dbs += (self.regularizer*bs)
        
        # Log additional info?
        if self.inspect:
            self.xs, self.ys = str(X[:, 1:]), str(ys[1:])
            self.scores, self.probs = scores, probs
            self.loss = loss
            self.dWhh, self.dbhh, self.dWxh, self.dbxh = dWhh, dbhh, dWxh, dbxh
            self.dWs, self.dbs = dWs, dbs
            self.hiddens = hiddens
            self.dhiddens = dhiddens
            self.dhiddens_local, self.dhiddens_downstream = dhiddens_local, dhiddens_downstream
        
        return State(loss, Gradients(dWhh, dbhh, dWxh, dbxh, dWs, dbs), hiddens[rollout])
    
    def learn(self):
        """Learn from a minibatch of training examples
        
        Run gradient descent on these examples
        
        """
        hidden = self.hidden
        loss, grads, new_hidden = self.forward_backward_prop()

        self.gradient_check(grads, hidden=hidden)
        self.hidden = new_hidden

        # Clip gradients
        #
        # Hidden and input weight gradients
        dWhh, dbhh = np.clip(grads.dWhh, -5, 5), np.clip(grads.dbhh, -5, 5)
        dWxh, dbxh = np.clip(grads.dWxh, -5, 5), np.clip(grads.dbxh, -5, 5) 

        # Softmax weight gradients
        dWs, dbs = np.clip(grads.dWs, -5, 5), np.clip(grads.dbs, -5, 5) 

        # Update weights
        #
        # Hidden and input weights
        self.Whh = self.Whh - self.learning_rate*dWhh
        self.bhh = self.bhh - self.learning_rate*dbhh
        self.Wxh = self.Wxh - self.learning_rate*dWxh
        self.bxh = self.bxh - self.learning_rate*dbxh
        
        # Softmax weights
        self.Ws = self.Ws - self.learning_rate*dWs
        self.bs = self.bs - self.learning_rate*dbs
        
        # Update batch index so the next time the next batch in line is used
        self.train_index = (self.train_index+self.rollout) % self.T
        
    def gradient_check(self, analytic_grads, hidden):
        """Verify gradient correctness
        
        The analytic dWhh, dbhh, dWxh, dbxh, dWs, and dbs come from doing forward-backward
        prop just a second ago. We numerically estimate these gradients on
        the *same* minibatch the analytic gradients were computed from and
        compare them to see if they are close.
        
        Note the same rollout is being used because this function gets
        called *before* the update to batch_index
        
        """
        if not self.gradient_checking:
            return
        
        numerical_grads = self.numerical_gradients(hidden)

        # Compute relative error
        #
        # Hidden and input differences
        for ana_grad, num_grad in zip(analytic_grads, numerical_grads):
            relative_errors = np.abs(num_grad-ana_grad) / (np.abs(num_grad) + np.abs(ana_grad))
            if np.linalg.norm(relative_errors) > 1e-5:
                warn('Gradient check failed!')

    def numerical_gradients(self, hidden):
        """Compute numerical gradients of f with respect the hidden, input, and
        softmax weights

        Returns numerical approximations of these

        """
        step = 1e-5
        params = {'Whh':self.Whh, 'bhh':self.bhh,
                'Wxh':self.Wxh, 'bxh':self.bxh, 'Ws':self.Ws, 'bs':self.bs}

        gradients = {'dWhh':np.zeros_like(self.Whh), 'dbhh':np.zeros_like(self.bhh),
                'dWxh':np.zeros_like(self.Wxh), 'dbxh':np.zeros_like(self.bxh),
                'dWs':np.zeros_like(self.Ws), 'dbs':np.zeros_like(self.bs)}

        for pname, param in params.items():
            h = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                ix = it.multi_index
                h[ix] = step
                
                gradients['d'+pname][ix] = (self.forward_backward_prop(**{pname:param+h}, hidden=hidden).loss -
                        self.forward_backward_prop(**{pname:param-h}, hidden=hidden).loss) / (2*step)

                h[ix] = 0
                it.iternext()
            
        return Gradients(**gradients)

    @property
    def info(self):
        """Get a snapshot of the model's most recent activity"""
        
        return Snapshot(self.xs, self.ys,
                        self.hiddens,
                        self.Whh, self.bhh, self.Wxh, self.bxh, self.Ws, self.bs,
                        self.dWhh, self.dbhh, self.dWxh, self.dbxh, self.dWs, self.dbs,
                        self.dhiddens, self.dhiddens_local, self.dhiddens_downstream,
                        self.scores, self.loss)
