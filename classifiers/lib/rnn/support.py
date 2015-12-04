from collections import namedtuple


State = namedtuple('State', ['loss', 'grads', 'hidden'])
Gradients = namedtuple('Gradients', ['dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs'])

Snapshot = namedtuple('State', ['xs', 'ys', 'hiddens', 'Whh', 'bhh', 'Wxh', 'bxh', 'Ws', 'bs', 'dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs', 'dhiddens', 'dhiddens_local', 'dhiddens_downstream', 'scores', 'loss'])
