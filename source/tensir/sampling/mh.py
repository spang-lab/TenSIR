import logging
import os
import pickle
import time

import numpy as np

from tensir.bounds_util import inside_bounds
from tensir.uniformization import forward


def _mh_one_iteration(data, theta, prior_bounds, ll, v, threads):
    theta_new = np.random.normal(theta, np.sqrt(v), size=2)

    if inside_bounds(np.exp(theta), prior_bounds):
        ll_new = forward.log_likelihood_dataset(data, np.exp(theta_new), threads=threads)
    else:
        ll_new = -np.inf
    alpha = min(1, np.exp(ll_new - ll))  # likelihood is being held logarithmically

    logging.info(f"Theta: {np.exp(theta_new)}, ll={ll_new}")

    if np.random.random() < 1 - alpha:  # discard with probability 1 - alpha
        logging.info(f"Discarded")
        theta_new = None

    return theta_new, ll_new, alpha


def metropolis_hastings_iterator(data, Theta0, prior_bounds, v, N=None, threads=1, load_state=None, save_state=None):
    """
    Run Metropolis-Hastings simulation on data under the SIR model, starting at `Theta0`.
    This function acts as a generator that yields the next `(log(alpha), log(beta))` point.
    Per default it runs indefinitely, except `N` is set.

    Supports loading/saving the internal state to resume a run at the last generated (accepted or unaccepted) point.

    Reference: https://arxiv.org/abs/2006.16194

    :param data: Numpy array with columns t, S, I, R
    :param Theta0: Initial (alpha, beta) of the SIR model
    :param prior_bounds: Bounds of the uniform prior (non-logarithmic)
    :param v: Variance for the q distribution for random walk MH
    :param N: Number of iterations. Set None to run indefinitely
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads if threads > 1)
    :param load_state: Path to a file where to load a generator state. Setting this will replace Theta0.
    :param save_state: Path to a file where to save the generator state after each generated point
    """
    if load_state is not None and os.path.exists(load_state):
        with open(load_state, "rb") as f:
            theta, ll, numpy_state = pickle.load(f)
        np.random.set_state(numpy_state)

    else:
        theta = np.log(Theta0)
        yield theta
        if not inside_bounds(Theta0, prior_bounds):
            raise ValueError(f"Theta0 {Theta0} is not in prior bounds {prior_bounds}")
        ll = forward.log_likelihood_dataset(data, Theta0, threads=threads)

    logging.info("Start")
    start = time.time()

    accepted_count = 1
    alphas = []

    t = 0
    while t != N:  # works if N is None
        logging.info(f"t={t}")

        theta_new, ll_new, alpha = _mh_one_iteration(data, theta, prior_bounds, ll, v, threads)
        logging.info(f"alpha={alpha:.5f}")
        alphas.append(alpha)
        logging.info(f"Accepted points: {accepted_count}, "
                     f"Average MH alpha: {sum(alphas) / len(alphas):.3f}, "
                     f"{(time.time() - start) / (t + 1):.2f} s/point")

        if theta_new is None:  # point proposal got rejected
            yield theta

        else:  # point proposal got accepted
            accepted_count += 1

            yield theta_new
            theta = theta_new
            ll = ll_new

        if save_state is not None:
            with open(save_state, "wb") as f:
                pickle.dump((theta, ll, np.random.get_state()), f)

        t += 1


def metropolis_hastings_fixed_count(data, Theta0, prior_bounds, v, count, threads=1, save_intermediate=None,
                                    load_state=None, save_state=None):
    """
    Run Metropolis-Hastings simulation on data under the SIR model, starting at `Theta0`.
    Returns a list of `count` points of `(log(alpha), log(beta))` parameters.

    Supports loading/saving the internal state to resume a run at the last generated (accepted or unaccepted) point.

    Reference: https://arxiv.org/abs/2006.16194

    :param data: Numpy array with columns t, S, I, R
    :param Theta0: Initial (alpha, beta) of the SIR model
    :param prior_bounds: Bounds of the uniform prior (non-logarithmic)
    :param v: Variance for the q distribution for random walk MH
    :param count: Number of points to output
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads if threads > 1)
    :param save_intermediate: Path to a csv file where to save intermediate points. Loads points from this file if it
                              exists.
    :param load_state: Path to a file where to load a generator state. Setting this will replace Theta0.
    :param save_state: Path to a file where to save the generator state after each generated point
    :return: List of `(log(alpha), log(beta))` parameters
    """

    if save_intermediate is not None and os.path.exists(save_intermediate):
        with open(save_intermediate, "r") as f:
            lines = f.readlines()
        points = [[float(l) for l in line.strip().split(",")] for line in lines]
        logging.info(f"{len(points)} points loaded")

    else:
        points = []

    for point in metropolis_hastings_iterator(data, Theta0, prior_bounds, v, N=None, threads=threads,
                                              load_state=load_state, save_state=save_state):

        if save_intermediate is not None:
            with open(save_intermediate, "a") as f:
                f.write(f"{point[0]},{point[1]}\n")

        points.append(point)

        if len(points) >= count:
            return points
