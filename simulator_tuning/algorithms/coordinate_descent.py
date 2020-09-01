import numpy as np
from typing import Callable


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
