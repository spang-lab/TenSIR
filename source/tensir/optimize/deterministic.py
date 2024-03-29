import numpy as np
from scipy.integrate import odeint


def solve_sir(SIR0, Theta, duration, steps):
    """
    Deterministically solves the SIR differential equations.

    :param SIR0: The initial SIR state as a triplet.
    :param Theta: (alpha, beta) tuple
    :param duration: The duration in days
    :param steps: Number of steps for the output
    :return: Solution values
    """
    S0, I0, R0 = SIR0
    N = np.sum(SIR0)
    alpha, beta = Theta

    def sir_model(yi, t):
        Si, Ii = yi
        return (-beta * Ii * Si / N,  # = dS/dt
                beta * Ii * Si / N - alpha * Ii)  # = dI/dt

    y0 = (S0, I0)  # initial vector
    t = np.linspace(0, duration, steps)  # time space
    SI = odeint(sir_model, y0, t)  # solve, transpose
    return np.column_stack((t, SI, N - np.sum(SI, axis=1)))
