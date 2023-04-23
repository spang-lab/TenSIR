import logging
import os.path
import os.path
import pickle
import time

import numpy as np

from tensir.uniformization import derivative, forward


def _leapfrog(data, theta, p, M_i, e, grad, threads):
    if grad is None:  # take previous gradient if available
        grad, ll = derivative.log_likelihood_gradient_dataset_cached(data, np.exp(theta), threads)
        logging.info(f"      Theta: {np.exp(theta)}, Gradient: {grad}, LL: {ll}, p: {p}")
        if np.any(grad == np.inf):  # exit if gradient = inf
            return theta, p, ll, grad
    p = p + e * 0.5 * grad
    theta = theta + e * np.dot(M_i, p)
    grad, ll = derivative.log_likelihood_gradient_dataset_cached(data, np.exp(theta), threads)
    logging.info(f"      Theta: {np.exp(theta)}, Gradient: {grad}, LL: {ll}, p: {p}")
    p = p + e * 0.5 * grad
    return theta, p, ll, grad


def _hmc_iter(data, theta, ll, M, M_i, e, L, threads=1):
    p = np.random.multivariate_normal(np.zeros(2), M)

    theta_new, p_new, ll_new, grad_new = theta, p, None, None
    for i in range(L):
        logging.info(f"  i={i}, ll={ll if ll_new is None else ll_new}")
        theta_new, p_new, ll_new, grad_new = _leapfrog(data, theta_new, p_new, M_i, e, grad_new, threads)
        if np.any(grad_new == np.inf):
            logging.warning("Gradient has inf, aborting leapfrogs")
            break

    # already calculated in leapfrog as a by-product of derivative.log_likelihood_gradient_dataset
    # ll_new = forward.log_likelihood_dataset(data, np.exp(theta_new), threads=threads)

    a_new = ll_new - 0.5 * np.dot(np.dot(p_new, M_i), p_new)
    a = ll - 0.5 * np.dot(np.dot(p, M_i), p)
    alpha = min(1, np.exp(a_new) / np.exp(a))

    if a_new == -np.inf or np.any(grad_new == np.inf):
        alpha = 0.

    if np.random.random() < 1 - alpha:  # discard with probability 1 - alpha
        logging.info(f"Discarded")
        theta_new = None

    return theta_new, ll_new, alpha


def hamilton_monte_carlo_iterator(data, Theta0, M, e, L, N=None, threads=1, load_state=None, save_state=None):
    """
    Run Hamiltonian Monte Carlo simulation on data under the SIR model, starting at `Theta0`.
    This function acts as a generator that yields the next `(log(alpha), log(beta))` point.
    Per default it runs indefinitely, except `N` is set.

    Supports loading/saving the internal state to resume a run at the last generated (accepted or unaccepted) point.

    :param data: Numpy array with columns t, S, I, R
    :param Theta0: Initial (alpha, beta) of the SIR model
    :param M: Covariance matrix of the HMC
    :param e: Epsilon parameter of the HMC
    :param L: L parameter of the HMC
    :param N: Number of iterations (Does not need to be the final number of points, since some are discarded!). Set None
              to run indefinitely
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads)
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
        ll = forward.log_likelihood_dataset(data, Theta0, threads=threads)

    logging.info("Start")
    start = time.time()

    M_i = np.linalg.inv(M)

    accepted_count = 1
    alphas = []

    t = 0
    while t != N:
        logging.info(f"t={t}")

        theta_new, ll, alpha = _hmc_iter(data, theta, ll, M, M_i, e, L, threads=threads)

        logging.info(f"alpha={alpha:.5f}")
        alphas.append(alpha)
        logging.info(f"Accepted points: {accepted_count}, "
                     f"Average HMC alpha: {sum(alphas) / len(alphas):.3f}, "
                     f"{(time.time() - start) / (t + 1):.2f} s/point")

        if theta_new is not None:
            accepted_count += 1

            yield theta_new
            theta = theta_new

        if save_state is not None:
            with open(save_state, "wb") as f:
                pickle.dump((theta, ll, np.random.get_state()), f)

        t += 1


def hamiltonian_monte_carlo_fixed_count(data, Theta0, M, e, L, count, threads=1, save_intermediate=None,
                                        load_state=None, save_state=None):
    """
    Run Hamiltonian Monte Carlo simulation on data under the SIR model, starting at `Theta0`.
    Returns a list of `count` points of `(log(alpha), log(beta))` parameters.

    Supports loading/saving the internal state to resume a run at the last generated (accepted or unaccepted) point.

    :param data: Numpy array with columns t, S, I, R
    :param Theta0: Initial (alpha, beta) of the SIR model
    :param M: Covariance matrix of the HMC
    :param e: Epsilon parameter of the HMC
    :param L: L parameter of the HMC
    :param count: Number of points to output
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads)
    :param save_intermediate: Path to a csv file where to save intermediate points. Loads points from this file if it
                              exists.
    :param load_state: Path to a file where to load a generator state. Setting this will replace Theta0.
    :param save_state: Path to a file where to save the generator state after each generated point
    :return: List of `(log(alpha), log(beta))` parameters
    """

    if os.path.exists(save_intermediate):
        with open(save_intermediate, "r") as f:
            lines = f.readlines()
        points = [[float(l) for l in line.strip().split(",")] for line in lines]
        logging.info(f"{len(points)} points loaded")

    else:
        points = []

    for point in hamilton_monte_carlo_iterator(data, Theta0, M, e, L, N=None, threads=threads, load_state=load_state,
                                               save_state=save_state):

        if save_intermediate is not None:
            with open(save_intermediate, "a") as f:
                f.write(f"{point[0]},{point[1]}\n")

        points.append(point)

        if len(points) >= count:
            return points
