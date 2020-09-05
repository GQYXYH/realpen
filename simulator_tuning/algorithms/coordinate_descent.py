"""Finding the optimal parameters for a given evaluation function with the Ray Tune Framework via coordinate descent.

Calculate the optimal parameters which maximize/minimize the given error/evaluation function by performing the
optimization parameter by parameter and holding on the residual parameters during each iteration.

Author: Moritz Schneider
"""

from typing import Callable, Dict

from ray import tune


def coordinate_descent(params: Dict, configuration: Dict, err_function: Callable, eps: float = 1e-1,
                       search_kwargs=None, analysis_kwargs=None, max_iters=100):
    """Performing the coordinate descent over the given parameters.

    Maximizing/minimizing the given error function parameter by parameter until the stopping threshold or
    the maximum iterations of coordinate descent are reached. In each iteration the error function is optimized
    parameter by parameter.

    To understand some of the arguments better it is encouraged to the documentation of Tune itself:
    https://docs.ray.io/en/master/tune/index.html

    Args:
        params: Parameters we want to optimize to fulfill the optimization goal.
        configuration: Configuration of the parameters for the search algorithm of Tune.
        err_function: Error/evaluation function we want to minimize/maximize.
        eps: Stopping threshold for the coordinate descent optimization (outer loop).
        search_kwargs: Arguments for the 'run' method of Tune.
        analysis_kwargs: Arguments for 'get_best_config' method of Tune. Optimization mode is defined here.
        max_iters: Maximum iterations to perform if the given threshold is not achieved.

    Returns:
        The optimized parameters which minimizes/maximizes the given error function.
    """
    err_history = []
    search_kwargs = search_kwargs if search_kwargs is not None else {}
    analysis_kwargs = analysis_kwargs if analysis_kwargs is not None else {"metric": "mean_accuracy"}

    err = err_function(params, use_tune=False)
    for j in range(max_iters):
        # Check if threshold is achieved
        if err < eps:
            break

        # Coordinate descent in vectorized form
        for x in params:
            param_config = {x: configuration[x]}
            params[x] = minimize(evaluation_function=err_function, configuration=param_config,
                                 search_kwargs=search_kwargs, analysis_kwargs=analysis_kwargs)

        # Calculate new error
        err = err_function(params, use_tune=False)
        print(f"Iteration {j + 1} with error {err}.")
        err_history.append(err)
    else:
        raise RuntimeError(f"Stopping threshold '{eps}' for optimization not reached. Value of the last error is '{err}'")

    return params# , err_history


def minimize(evaluation_function: Callable, configuration: Dict, search_kwargs: Dict, analysis_kwargs: Dict):
    """Minimizing/maximizing the given evaluation function for the parameter which is defined in the configuration.

    Args:
        evaluation_function: Error function we want to minimize/maximize.
        configuration: Search space dictionary for the parameter we want to optimize.
        search_kwargs: Arguments for tune.run.
        analysis_kwargs: Arguments for get_best_config of tune.

    Returns:
        The new recommendation for the parameter.
    """
    if "search_alg" in search_kwargs:
        search_kwargs["search_alg"] = search_kwargs["search_alg"](**analysis_kwargs)
    analysis = tune.run(evaluation_function, config=configuration, **search_kwargs)
    recommendation = analysis.get_best_config(**analysis_kwargs)
    return recommendation
