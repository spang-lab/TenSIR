import logging
import os
import sys
from os.path import join

import numpy as np

from paths import AUSTRIA_DATA_CACHE_PATH, AUSTRIA_MONTHLY_HMC_POINTS_DIR, LOGS_DIR, HMC_STATES_DIR
from tensir import data
from tensir.optimize import hamiltonian_monte_carlo_fixed_count


def main():
    month, run = int(sys.argv[1]), int(sys.argv[2])
    points_per_run = 100
    seed = month * 1000 + run
    threads = 48

    log_dir = join(LOGS_DIR, f"{month:02d}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = join(log_dir, f"hmc-{month:02d}-{run:02d}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a",
                        format="[%(asctime)s %(levelname)s] %(message)s")

    out_dir = join(AUSTRIA_MONTHLY_HMC_POINTS_DIR, f"{month:02d}")
    intermediate_dir = join(AUSTRIA_MONTHLY_HMC_POINTS_DIR, "intermediate", f"{month:02d}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(HMC_STATES_DIR, exist_ok=True)

    data_sir = data.covid19_austria_daily(start=f"2020-{month:02d}-01",
                                          end=f"2020-{month + 1:02d}-01",
                                          cache_path=AUSTRIA_DATA_CACHE_PATH)

    np.random.seed(seed)
    name = f"hmc-points-{month:02d}-{run:02d}"
    csv_name = f"{name}.csv"
    state_path = join(HMC_STATES_DIR, f"{name}.pkl")

    points = tensir.optimize.hamiltonian_monte_carlo_fixed_count(data_sir, Theta0=(0.1, 0.1),
                                                                 M=np.diag(np.ones(2) * 2), e=0.05, L=5,
                                                                 count=points_per_run, threads=threads,
                                                                 save_intermediate=join(intermediate_dir, csv_name),
                                                                 load_state=state_path, save_state=state_path)
    filepath = join(out_dir, csv_name)

    with open(filepath, "w") as f:
        for alpha, beta in points:
            f.write(f"{alpha},{beta}\n")


if __name__ == '__main__':
    main()
