import itertools
import os
from os.path import join

import numpy as np
import typer

from paths import AUSTRIA_DATA_CACHE_PATH, PROJECT_ROOT
from tensir import data
from tensir.uniformization.derivative import log_likelihood_gradient_dataset

landscape_points_dir = join(PROJECT_ROOT, "landscape")


def main(month: int = typer.Option(...),
         sector: int = typer.Option(...),
         threads: int = typer.Option(...)):
    s = 3
    size = 10
    space = np.geomspace(0.001, 1., s * size)
    alphas = space[(sector % s) * size:(sector % s + 1) * size]
    betas = space[(sector // s) * size:(sector // s + 1) * size]

    data_sir = data.covid19_austria_daily(start=f"2020-{month:02d}-01",
                                          end=f"2020-{month + 1:02d}-01",
                                          cache_path=AUSTRIA_DATA_CACHE_PATH)

    os.makedirs(landscape_points_dir, exist_ok=True)

    path = join(landscape_points_dir, f"{month:02d}-{sector:02d}.csv")

    with open(path, "w") as f:
        f.write("alpha,beta,d_alpha,d_beta,ll\n")

    points = list(itertools.product(alphas, betas))
    np.random.shuffle(points)

    for alpha, beta in points:
        (d_alpha, d_beta), ll = log_likelihood_gradient_dataset(data_sir, (alpha, beta), threads=threads)

        with open(path, "a") as f:
            f.write(f"{alpha},{beta},{d_alpha},{d_beta},{ll}\n")


if __name__ == "__main__":
    typer.run(main)
