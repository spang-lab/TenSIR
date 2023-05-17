import itertools
import logging
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


def ess_over_time(hparams: tuple, func: callable) -> Optional[float]:
    np.random.seed(SEED)

    data_sir = data.covid19_austria_daily(start=f"2020-06-01",
                                          end=f"2020-07-01",
                                          cache_path=AUSTRIA_DATA_CACHE_PATH)

    iterator = func(data_sir, Theta0, *hparams, threads=THREADS)

    # get one point to ensure initialization of all internal variables
    next(iterator)
    start = time.perf_counter()

    alphas, betas = np.zeros(0), np.zeros(0)
    for p in iterator:
        alpha, beta = p
        alphas = np.append(alphas, alpha)
        betas = np.append(betas, beta)

        ess_alpha = arviz.ess(alphas, method="folded")
        ess_beta = arviz.ess(betas, method="folded")

        logging.warning(f"ESS alpha: {ess_alpha}, ESS beta: {ess_beta}, wtime: {time.perf_counter() - start:.1f}s")

        if ess_alpha > TARGET_ESS and ess_beta > TARGET_ESS:
            break

        if time.perf_counter() - start > 100:
            logging.critical(f"Aborting with ESS alpha: {ess_alpha}, ESS beta: {ess_beta}")
            return None

    return time.perf_counter() - start


def find_mh_params():
    results = []
    for v in np.geomspace(0.001, 0.3, 30):
        logging.critical(f"Measuring time to target ess with {v=}")
        results.append((v, ess_over_time((v,), metropolis_hastings_iterator)))

        print(results)


def find_hmc_params():
    results = []
    for m, e, L in itertools.product((0.5, 1.0, 2.0),
                                     np.linspace(0.1, 0.5, 4),
                                     (3, 5)):
        logging.critical(f"Measuring time to target ess with {m=}, {e=}, {L=}")
        M = np.eye(2) * m
        results.append(((m, e, L), ess_over_time((M, e, L), hamilton_monte_carlo_iterator)))

        print(results)


def main():
    logging.basicConfig(level=logging.CRITICAL, format="[%(asctime)s %(levelname)s] %(message)s")
    # find_mh_params()
    find_hmc_params()


if __name__ == "__main__":
    main()
