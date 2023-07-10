import itertools

import numpy as np

from tensir.optimize import deterministic


def _calculate_error(data, Theta, duration, steps):
    SIR0 = data[0, 1:]
    solution = deterministic.solve_sir(SIR0, Theta, duration, steps)

    SI_data = data[:, 1:3]
    SI_solved = solution[:, 1:3]

    return np.sum((SI_data - SI_solved) ** 2)


def grid_search_deterministic(data, Theta_min, Theta_max, resolution):
    """
    Find the Theta that best explains the data under the deterministic ODE model via grid search.

    :param data: The data
    :param Theta_min: Lower limit of the grid
    :param Theta_max: Upper limit of the grid
    :param resolution: Resolution per axis of the grid
    :return: The best Theta
    """
    duration = data[-1, 0] - data[0, 0]
    steps = data.shape[0]

    alpha_range, beta_range = zip(Theta_min, Theta_max)

    grid = list(itertools.product(np.logspace(*np.log10(alpha_range), resolution),
                                  np.logspace(*np.log10(beta_range), resolution)))

    min_error = np.inf
    best_Theta = None
    for Theta in grid:

        error = _calculate_error(data, Theta, duration, steps)
        if error < min_error:
            min_error = error
            best_Theta = Theta

    return best_Theta
