import numpy as np
from typing import Callable
from ray.tune.suggest import Searcher


class CoordinateDescent(Searcher):
    def __init__(self, metric="mean_loss", mode="min", **kwargs):
        super(CoordinateDescent, self).__init__(metric=metric, mode=mode, **kwargs)
        self.optimizer = Optimizer()
        self.configurations = {}

    def suggest(self, trial_id):
        configuration = self.optimizer.query()
        self.configurations[trial_id] = configuration

    def on_trial_complete(self, trial_id, result, **kwargs):
        configuration = self.configurations[trial_id]
        if result and self.metric in result:
            self.optimizer.update(configuration, result[self.metric])    

def coordinate_descent(params: np.Array, err_function: Callable, learning_rate: float = .03, eps: float = 1e-10):
    '''Coordinate gradient descent for linear regression'''
    # Initialisation of useful values
    m, n = params.shape
    eta = learning_rate
    err_history = []
    j = 0

    while True:
        # Coordinate descent in vectorized form
        # h = X @ params
        # gradient = (X[:, j] @ (h - y))
        err = err_function(params)
        err_history.append(err)

        if err < eps:
            break

        params[j] = params[j] - eta * err
        j = j + 1 if j < n else 0

    return params, err_history
