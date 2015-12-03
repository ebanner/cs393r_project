from collections import namedtuple


State = namedtuple('State', ['loss', 'dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs'])

Snapshot = namedtuple('State', ['xs', 'ys', 'Whh', 'bhh', 'Wxh', 'bxh', 'Ws', 'bs', 'dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs', 'dhiddens', 'dhiddens_local', 'dhiddens_downstream', 'scores', 'loss'])
