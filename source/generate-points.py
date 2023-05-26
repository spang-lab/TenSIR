import logging
import os
import sys
from enum import Enum
from os.path import join

import numpy as np
import typer

import tensir
from paths import AUSTRIA_DATA_CACHE_PATH, HMC_STATES_DIR, MH_LOGS_DIR, \
    HMC_LOGS_DIR, MH_POINTS_DIR, HMC_POINTS_DIR, MH_STATES_DIR
from tensir.bounds_util import inside_bounds
from tensir.sampling.hmc import hamiltonian_monte_carlo_fixed_count
from tensir.sampling.mh import metropolis_hastings_fixed_count
from tensir.uniformization import forward


class Sampling(str, Enum):
    HMC = "hmc"
    MH = "mh"


def draw_Theta0(
        data_sir: np.ndarray, prior_bounds: tuple[float, float], mu: float, std: float, threads: int
) -> np.ndarray:
    logging.info("Drawing Theta0")
    while True:
        Theta0 = np.exp(np.random.normal(loc=mu, scale=std, size=2))
        logging.info(f"Candidate Theta0: {Theta0}")

        inside = inside_bounds(Theta0, prior_bounds)
        logging.info(f"Inside bounds? {inside}")

        finite = np.isfinite(forward.log_likelihood_dataset(data_sir, Theta0, threads=threads))
        logging.info(f"Finite LL? {finite}")

        if inside and finite:
            return Theta0


def main(
        sampling: Sampling = typer.Option(...),
        points: int = typer.Option(...),
        month: int = typer.Option(...),
        run: int = typer.Option(...),
        threads: int = typer.Option(...),
        hmc_m: float = 2.,
        hmc_e: float = 0.05,
        hmc_l: int = 5,
        mh_v: float = 0.1
):
    seed = month * 1000 + run

    if sampling == Sampling.HMC:
        log_dir = HMC_LOGS_DIR
        points_dir = HMC_POINTS_DIR
        states_dir = HMC_STATES_DIR

    elif sampling == Sampling.MH:
        log_dir = MH_LOGS_DIR
        points_dir = MH_POINTS_DIR
        states_dir = MH_STATES_DIR

    else:
        raise ValueError(f"Illegal value '{sampling}' for sampling argument")

    log_month_dir = join(log_dir, f"{month:02d}")
    os.makedirs(log_month_dir, exist_ok=True)
    log_file = join(log_month_dir, f"{sampling}-{month:02d}-{run:02d}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a",
                        format="[%(asctime)s %(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    out_dir = join(points_dir, f"{month:02d}")
    intermediate_dir = join(points_dir, "intermediate", f"{month:02d}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    data_sir = tensir.data.covid19_austria_daily(start=f"2020-{month:02d}-01",
                                                 end=f"2020-{month + 1:02d}-01",
                                                 cache_path=AUSTRIA_DATA_CACHE_PATH)

    np.random.seed(seed)
    name = f"{sampling}-points-{month:02d}-{run:02d}"
    csv_name = f"{name}.csv"
    state_path = join(states_dir, f"{name}.pkl")

    logging.info(f"Hyperparameters: {hmc_m=}, {hmc_e=}, {hmc_l=}, {mh_v=}")

    prior_bounds = (0.01, 1.)

    # parameters are in the log normally distributed with a mean of about 0.05
    Theta0 = draw_Theta0(data_sir, prior_bounds, mu=-3, std=0.2, threads=threads)

    if sampling == Sampling.HMC:
        points = hamiltonian_monte_carlo_fixed_count(data_sir, Theta0=Theta0, prior_bounds=prior_bounds,
                                                     M=np.eye(2) * hmc_m, e=hmc_e, L=hmc_l,
                                                     count=points, threads=threads,
                                                     save_intermediate=join(intermediate_dir, csv_name),
                                                     load_state=state_path, save_state=state_path)
    elif sampling == Sampling.MH:
        points = metropolis_hastings_fixed_count(data_sir, Theta0=Theta0, prior_bounds=prior_bounds, v=mh_v,
                                                 count=points, threads=threads,
                                                 save_intermediate=join(intermediate_dir, csv_name),
                                                 load_state=state_path, save_state=state_path)
    else:
        raise ValueError(f"Illegal value '{sampling}' for sampling argument")

    filepath = join(out_dir, csv_name)

    with open(filepath, "w") as f:
        for alpha, beta in points:
            f.write(f"{alpha},{beta}\n")


if __name__ == '__main__':
    typer.run(main)
