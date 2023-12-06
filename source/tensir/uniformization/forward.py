from multiprocessing import Pool

import numpy as np

from tensir.uniformization._tensormath import shuffle_diag

np.seterr(divide="ignore", under="ignore", invalid="ignore")  # suppress log(0), exp(-inf), log(-x) warnings


def _uniformization(args):
    data, Theta = args

    # separate time from SIR, so SIR can be converted to int and used for indexation
    data_t = data[:, 0]
    data_SIR = np.int64(data[:, 1:])

    # assume N constant
    N = np.sum(data_SIR[0])

    alpha, beta = Theta

    # unpack the data
    t0 = data_t[0]
    S0, I0, R0 = data_SIR[0]

    t1 = data_t[1]
    S1, I1, R1 = data_SIR[1]

    t = t1 - t0

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

    # set q initial state to 100%
    pt = np.zeros(state_count)
    q = np.zeros(state_count)
    q[state0] = 1.

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

        # activate breaking if increment at the relevant index was not zero once
        if inc[state1] > 0.:
            was_not_zero = True

        # break early if the increment is zero again
        if was_not_zero and inc[state1] <= 0.:
            break

        # n <- n + 1
        n += 1

        # q <- P * q
        # = (I + Q / gamma) * q
        # = (I + S_p_inf / gamma ⊗ I_p_inf + ...) * q
        # = q + v1 + v2 + v3 + v4
        v1 = shuffle_diag(S_p_inf * beta / N / gamma, 1, I_p_inf, -1, q)
        v2 = shuffle_diag(S_p_rec * alpha / gamma, 0, I_p_rec, 1, q)
        v3 = shuffle_diag(S_n_inf * beta / N / gamma, 0, I_n_inf, 0, q)
        v4 = shuffle_diag(S_n_rec * alpha / gamma, 0, I_n_rec, 0, q)
        q = q + v1 + v2 + v3 + v4

        # w <- gamma * t / n * w
        w_log = np.log(gamma * t / n) + w_log

    likelihood = pt[state1]
    if likelihood < 0.:  # numerical imprecisions
        likelihood = 0.

    return np.log(likelihood)


def log_likelihood_dataset(data, theta, theta_is_log, threads=1):
    """
    Calculate the log-likelihood of parameter `Theta` on `data` under the SIR model.

    :param data: Numpy array with columns t, S, I, R
    :param theta: [alpha, beta] of the SIR model
    :param theta_is_log: Whether theta is logarithmic (i.e. [log(alpha), log(beta)])
    :param threads: Run the code in parallel
    :return: Log-likelihood
    """

    if theta_is_log:
        theta = np.exp(theta)

    args = ((data[k:k + 2], theta) for k in range(data.shape[0] - 1))

    if threads > 1:
        with Pool(threads) as pool:
            sub_lls = tuple(pool.map(_uniformization, args, chunksize=1))

    else:
        sub_lls = tuple(map(_uniformization, args))

    ll = np.mean(sub_lls)

    return ll
