import numpy as np


def gillespie(SIR0, Theta, duration, steps):
    alpha, beta = Theta
    S, I, R = SIR0
    N = S + I + R
    steptime = duration / steps

    data = np.zeros((steps, 4))

    # t
    data[:, 0] = np.linspace(0., duration, steps)

    data[0, 1] = S
    data[0, 2] = I
    data[0, 3] = R

    t_frac = 0.
    s = 1
    while s < steps:

        while t_frac < steptime:

            w0 = beta * S * I / N
            w1 = alpha * I
            weight_sum = w0 + w1

            if weight_sum > 0:
                if np.random.random() < w0 / weight_sum:
                    S -= 1
                    I += 1
                else:
                    I -= 1
                    R += 1

                dt = np.random.exponential(1 / weight_sum)
                t_frac += dt
            else:
                t_frac = duration - (s - 1) * steptime

        for i in range(int(t_frac / steptime)):
            if s < steps:
                data[s, 1] = S
                data[s, 2] = I
                data[s, 3] = R
                s += 1
            else:
                break

        t_frac = t_frac % steptime

    return data
