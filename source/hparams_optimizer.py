import itertools
import logging
import math
import sys
import time
from typing import Optional

import arviz
import numpy as np

from paths import AUSTRIA_DATA_CACHE_PATH
from tensir import data
from tensir.sampling.hmc import hamilton_monte_carlo_iterator
from tensir.sampling.mh import metropolis_hastings_iterator

SEED = 1
THREADS = 32
TARGET_ESS = 100
Theta0 = (0.1, 0.1)
prior_bounds = (0.01, 1.)
burnin_frac = 0.1


def ess_over_time(hparams: tuple, func: callable) -> Optional[float]:
    np.random.seed(SEED)

    data_sir = data.covid19_austria_daily(start=f"2020-06-01",
                                          end=f"2020-07-01",
                                          cache_path=AUSTRIA_DATA_CACHE_PATH)

    iterator = func(data_sir, Theta0, prior_bounds, *hparams, threads=THREADS)

    # get one point to ensure initialization of all internal variables
    next(iterator)
    start = time.perf_counter()

    alphas, betas = np.zeros(0), np.zeros(0)
    for p in iterator:
        alpha, beta = p
        alphas = np.append(alphas, alpha)
        betas = np.append(betas, beta)

        burnin = math.ceil(len(alphas) * burnin_frac)

        ess_alpha = arviz.ess(alphas[burnin:], method="folded")
        ess_beta = arviz.ess(betas[burnin:], method="folded")

        logging.warning(f"ESS alpha: {ess_alpha}, ESS beta: {ess_beta}, wtime: {time.perf_counter() - start:.1f}s")

        if ess_alpha > TARGET_ESS and ess_beta > TARGET_ESS:
            break

        if time.perf_counter() - start > 300:
            logging.critical(f"Aborting with ESS alpha: {ess_alpha}, ESS beta: {ess_beta}")
            return None

    return time.perf_counter() - start


def find_mh_params():
    results = []
    for v in np.geomspace(0.001, 0.3, 30):
        logging.critical(f"Measuring time to target ess with {v=}")
        t = ess_over_time((v,), metropolis_hastings_iterator)
        if t is not None:
            results.append((v, t))
            logging.info(results)

    logging.info(results)


def find_hmc_params():
    results = []
    for m, e, L in itertools.product(np.linspace(0.5, 2.0, 4),
                                     np.geomspace(0.001, 0.2, 6),
                                     (3, 5, 7, 10)):
        logging.critical(f"Measuring time to target ess with {m=}, {e=}, {L=}")
        M = np.eye(2) * m
        t = ess_over_time((M, e, L), hamilton_monte_carlo_iterator)
        if t is not None:
            results.append(((m, e, L), t))
            logging.info(results)

    logging.info(results)


def main():
    logging.basicConfig(level=logging.WARNING, filename="hparams_grid_search.log", filemode="w",
                        format="[%(asctime)s %(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # find_mh_params()
    find_hmc_params()


if __name__ == "__main__":
    main()
