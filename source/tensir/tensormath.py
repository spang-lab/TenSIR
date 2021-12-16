def _matmul_diag(W, values, k):
    """
    Performs efficient matrix multiplication `A * W`, where `A` is a `k`-th diagonal matrix with entries `values`.
    """

    if k > 0:
        W[:-k, :] = W[k:, :] * values.reshape((-1, 1))
        W[-k:, :] = 0.
        return W.T
    elif k < 0:
        k = -k
        W[k:, :] = W[:-k, :] * values.reshape((-1, 1))
        W[:k, :] = 0.
        return W.T
    else:
        return W.T * values


def shuffle_diag(values_A, k_A, values_B, k_B, v):
    """
    Calculates (A ⊗ B) v for square k-th diagonal matrices A, B (represented by values_A, values_B respectively) and
    vector v.
    """

    N_A = values_A.size + abs(k_A)
    N_B = values_B.size + abs(k_B)

    W = v.copy().reshape((N_A, -1))
    W = _matmul_diag(W, values_A, k_A)
    W = W.reshape((N_B, -1))
    W = _matmul_diag(W, values_B, k_B)
    return W.ravel()
