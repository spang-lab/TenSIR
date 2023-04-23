import logging

import numpy as np

from tensir.uniformization import derivative


def gradient_ascent(data, Theta0, lr, conv, threads=1):
    """
    Learns the parameter `theta` on `data` under the SIR model, starting at `theta0`.

    :param data: Numpy array with columns t, S, I, R
    :param Theta0: Initial (alpha, beta) of the SIR model
    :param lr: Learning rate
    :param conv: Converged if all(abs(old_theta - new_theta) <= conv)
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads)
    :return: Learned [alpha, beta]
    """

    Theta = Theta0
    old_Theta = np.array([np.inf, np.inf])
    while np.any(np.abs(old_Theta - Theta) > conv):
        gradient, ll = derivative.log_likelihood_gradient_dataset(data, Theta, threads=threads)
        old_Theta = Theta
        Theta = np.exp(np.log(Theta) + lr * gradient)
        logging.info(
            f"Theta: {old_Theta}, LL: {ll}, Theta_new: {Theta}, Gradient: {gradient}, abs(Delta): {np.abs(old_Theta - Theta)}")

    return Theta
