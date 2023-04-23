import logging
from multiprocessing import Pool

import numpy as np
from cachetools import cached
from cachetools.keys import hashkey

from tensir._pool import NestablePool
from tensir.uniformization._tensormath import shuffle_diag


def _differentiated_uniformization(args):
    param, implicit_tensors, max_factors, state_count, states, N, t, Theta, gamma = args

    # unpack
    S_p_inf, I_p_inf, S_p_rec, I_p_rec, S_n_inf, I_n_inf, S_n_rec, I_n_rec = implicit_tensors
    S_max, I_max = max_factors
    state0, state1 = states
    alpha, beta = Theta

    if param == 0:  # alpha
        gamma_d = I_max * alpha
    else:  # beta
        gamma_d = S_max * I_max * beta / N

    # set q initial state to 100%
    pt = np.zeros(state_count)
    pt_d = np.zeros(state_count)
    q = np.zeros(state_count)
    q[state0] = 1.
    q_d = np.zeros(state_count)

    n = 0
    w_log = 0.  # we hold w logarithmically because it gets large very quickly

    # early brake flag needs to be activated first
    was_not_zero = False

    double_eps = np.nextafter(0, 1)  # 5e-324
    n_max = gamma - 1 / 3 * np.log(double_eps) * (1 + np.sqrt(1 - 18 * gamma / np.log(double_eps))) - 1  # Sherlock 2020
    while n <= n_max:
        # p(t) <- p(t) + e^(-gamma * t) * w * q
        inc = np.exp(-gamma * t + w_log) * q
        pt = pt + inc

        # p(t)' <- p(t)' + e^(-gamma * t) * w * (q' + gamma' * (n / gamma - t) * q)
        #                                        |---------------temp-------------|
        temp = q_d + gamma_d * (n / gamma - t) * q
        pt_d = pt_d + np.sign(temp) * np.exp(-gamma * t + w_log + np.log(np.abs(temp)))

        # activate breaking if increment at the relevant index was not zero once
        if inc[state1] > 0.:
            was_not_zero = True

        # break early if the increment is zero again
        if was_not_zero and inc[state1] <= 0.:
            break

        # n <- n + 1
        n = n + 1

        # q' <- P' * q + P * q'
        # = (-gamma' / gamma^2 * Q + Q' / gamma) * q + (I + Q / gamma) * q'
        # = -gamma' / gamma^2 * Q * q   +   Q' / gamma * q   +   q'   +   Q / gamma * q'

        # Q * q / gamma
        Qq1 = shuffle_diag(S_p_inf * beta / N / gamma, 1, I_p_inf, -1, q)
        Qq2 = shuffle_diag(S_p_rec * alpha / gamma, 0, I_p_rec, 1, q)
        Qq3 = shuffle_diag(S_n_inf * beta / N / gamma, 0, I_n_inf, 0, q)
        Qq4 = shuffle_diag(S_n_rec * alpha / gamma, 0, I_n_rec, 0, q)

        # Q' * q / gamma
        if param == 0:  # alpha
            Q_q1 = shuffle_diag(S_p_rec * alpha / gamma, 0, I_p_rec, 1, q)
            Q_q2 = shuffle_diag(S_n_rec * alpha / gamma, 0, I_n_rec, 0, q)
        else:  # beta
            Q_q1 = shuffle_diag(S_p_inf * beta / N / gamma, 1, I_p_inf, -1, q)
            Q_q2 = shuffle_diag(S_n_inf * beta / N / gamma, 0, I_n_inf, 0, q)

        # Q * q' / gamma
        Qq_1 = shuffle_diag(S_p_inf * beta / N / gamma, 1, I_p_inf, -1, q_d)
        Qq_2 = shuffle_diag(S_p_rec * alpha / gamma, 0, I_p_rec, 1, q_d)
        Qq_3 = shuffle_diag(S_n_inf * beta / N / gamma, 0, I_n_inf, 0, q_d)
        Qq_4 = shuffle_diag(S_n_rec * alpha / gamma, 0, I_n_rec, 0, q_d)

        #     -gamma' / gamma^2 * Q * q                  + Q' / gamma * q + q' + Q / gamma * q'
        q_d = -gamma_d / gamma * (Qq1 + Qq2 + Qq3 + Qq4) + (Q_q1 + Q_q2) + q_d + (Qq_1 + Qq_2 + Qq_3 + Qq_4)
        #                     ^ second gamma already in QaX

        # a <- P * a
        # = (I + Q / gamma) * a
        q = q + Qq1 + Qq2 + Qq3 + Qq4

        # w <- gamma * t / n * w
        w_log = np.log(gamma * t / n) + w_log

    if pt[state1] <= 0.:  # numerical imprecisions, also define 0 / 0 := inf
        if pt_d[state1] != 0.:
            logging.warning("x/0 encountered: pt_d accumulated values, but pt didn't")
        return np.inf, 0.  # return inf gradient and 0 ll

    return pt_d[state1] / pt[state1], pt[state1]


def _log_likelihood_gradient(args):
    data, Theta, threads = args

    # separate time from SIR, so SIR can be converted to int and used for indexation
    t0 = data[0, 0]
    S0, I0, R0 = np.int64(data[0, 1:])

    t1 = data[1, 0]
    S1, I1, R1 = np.int64(data[1, 1:])

    t = t1 - t0

    # assume N constant
    N = int(np.sum(data[0, 1:]))

    alpha, beta = Theta

    # population changes
    dS = S1 - S0  # always negative, since S only gets smaller over time
    dR = R1 - R0  # always positive, since R only gets bigger over time

    # S and I could have only been in these intervals [min, max]
    S_min = S1  # S1 must be min
    S_max = S0  # S0 must be max

    I_min = I0 - dR  # min value for I if first all dR recovered
    I_min = max(0, I_min)  # but can't be less than 0
    I_max = I0 - dS  # max value for I if first all dS got infected
    I_max = min(N, I_max)  # but can't be more than N

    # interval lengths
    S_len = S_max - S_min + 1  # plus 1 because the max value is also included
    I_len = I_max - I_min + 1  # same applies here

    # state indices in Q and p resp. according to these intervals
    # the "block" is defined by S, inside each block the index is defined by I
    # initial state
    block_index = S_len - 1  # must lie in last block
    inside_index = I0 - I_min  # subtract minimum I to get index
    state0 = block_index * I_len + inside_index  # block size is I_len

    # end state
    block_index = 0  # must lie in first block
    inside_index = I1 - I_min  # subtract minimum I to get index
    state1 = block_index * I_len + inside_index  # block size is I_len

    # number of possible states
    state_count = S_len * I_len

    # implicit sub-matrices, p = positive, n = negative

    # first +1 because superdiag, second +1 because upper bound excluded in numpy.arange
    S_p_inf = np.arange(S_min + 1, S_max + 1)
    # -1 because subdiag, +1 because upper bound excluded in numpy.arange
    I_p_inf = np.arange(I_min, I_max - 1 + 1)
    # only ones
    S_p_rec = np.ones(S_len)
    # first +1 because superdiag, second +1 because upper bound excluded in numpy.arange
    I_p_rec = np.arange(I_min + 1, I_max + 1)

    # +1 because upper bound excluded in numpy.arange
    S_n_inf = -np.arange(S_min, S_max + 1)
    # +1 because upper bound excluded in numpy.arange
    I_n_inf = np.arange(I_min, I_max + 1)
    if I_max == N:
        I_n_inf[-1] = 0.
    # only ones
    S_n_rec = -np.ones(S_len)
    # +1 because upper bound excluded in numpy.arange
    I_n_rec = np.arange(I_min, I_max + 1)

    gamma = S_max * I_max * beta / N + I_max * alpha  # upper bound on last diagonal entry

    implicit_tensors = (S_p_inf, I_p_inf, S_p_rec, I_p_rec, S_n_inf, I_n_inf, S_n_rec, I_n_rec)
    max_factors = (S_max, I_max)
    states = (state0, state1)
    args = tuple(
        (param, implicit_tensors, max_factors, state_count, states, N, t, Theta, gamma)
        for param in (0, 1))  # 0 = alpha, 1 = beta

    if threads > 1:
        with Pool(2) as pool:
            result = pool.map(_differentiated_uniformization, args, chunksize=1)

    else:
        result = map(_differentiated_uniformization, args)

    result = np.array(tuple(result))
    gradient = result[:, 0]
    ll = np.log(result[0, 1])  # likelihood equal for both params, take first one

    logging.debug(f"Calculated sub-gradient: {gradient}, LL: {ll}")

    return gradient, ll


def log_likelihood_gradient_dataset(data, Theta, threads=1):
    """
    Returns the mean gradient of the log-likelihood of a parameter `Theta` of a dataset under the SIR model.
    The mean gradient is the mean of all gradients between two consecutive rows in `data`.

    Also returns the mean log-likelihood-score as the averaged sum over all log-likelihoods at `Theta` which is
    calculated as a by-product.

    :param data: Numpy array with columns t, S, I, R
    :param Theta: [alpha, beta] of the SIR model
    :param threads: Run the code in parallel (will make use of floor(threads / 2) * 2 threads)
    :return: Gradient, log-likelihood-score
    """

    # threads argument is 2 for threads >= 2
    args = ((data[k:k + 2], Theta, min(threads, 2)) for k in range(data.shape[0] - 1))

    if threads > 1:
        with NestablePool(threads // 2) as pool:
            result = tuple(pool.map(_log_likelihood_gradient, args, chunksize=1))

    else:
        result = tuple(map(_log_likelihood_gradient, args))

    sub_gradients, lls = zip(*result)

    gradient = np.mean(sub_gradients, axis=0)
    ll = np.mean(lls)
    logging.debug(f"Calculated gradient: {gradient}, LL: {ll}")

    return gradient, ll


@cached(cache={}, key=lambda d, T, t: hashkey(tuple(map(tuple, d)), tuple(T)))
def log_likelihood_gradient_dataset_cached(data, Theta, threads):
    return log_likelihood_gradient_dataset(data, Theta, threads)
