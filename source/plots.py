import datetime as dt
import os
from enum import Enum
from functools import partial
from os.path import join

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import cm, dates, ticker
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import tensir
from paths import PLOTS_DIR, AUSTRIA_DATA_CACHE_PATH, AUSTRIA_TIMELINE_PLOT_PATH, HMC_POINTS_DIR, MH_POINTS_DIR, \
    HMC_PLOT_PATH, MH_PLOT_PATH, HMC_ESS_PATH, MH_ESS_PATH
from tensir import data

rcParams["text.usetex"] = True
sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "axes.edgecolor": "0.15",
                            "font.family": "serif", "font.serif": ["Computer Modern"]})


class Sampling(str, Enum):
    HMC = "hmc"
    MH = "mh"


app = typer.Typer()


@app.command()
def density(sampling: Sampling = typer.Option(..., ),
            output_stats: bool = False):
    levels = 5
    burn_in = 100  # number of points of each run to skip
    xlim = (0.01, 0.3)
    ylim = (0.01, 0.3)
    smooth = 3.5
    thresh = 0.02

    if sampling == "hmc":
        points_dir = HMC_POINTS_DIR
        plot_path = HMC_PLOT_PATH

    elif sampling == "mh":
        points_dir = MH_POINTS_DIR
        plot_path = MH_PLOT_PATH

    data_files = []
    for m in range(3, 9):
        month_dir = join(points_dir, f"{m:02d}")
        os.makedirs(month_dir, exist_ok=True)
        month_files = [join(month_dir, f) for f in os.listdir(month_dir)]
        data_files.append(month_files)

    titles = ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020", "August 2020"]

    grid_searched = [tensir.optimize.grid.grid_search_deterministic(
        data.covid19_austria_daily(f"2020-0{m}-01", f"2020-0{m + 1}-01", cache_path=AUSTRIA_DATA_CACHE_PATH),
        (xlim[0], ylim[0]), (xlim[1], ylim[1]), 40)
        for m in range(3, 9)]

    greys = cm.get_cmap("Greys", 100)
    cmap = ListedColormap(greys(np.linspace(0.2, 0.8)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)

    for i, (files, title, Theta_grid) in enumerate(zip(data_files, titles, grid_searched)):

        ax = axes[i // 3, i % 3]

        points = []
        skipped = 0
        for file in files:
            with open(file, "r") as f:
                data_lines = f.readlines()

            points = points + [[float(x) for x in row.strip().split(",")] for row in data_lines[burn_in:]]
            skipped += len(data_lines[:burn_in])

        if len(points) > 0:
            data_x, data_y = zip(*points)
        else:
            data_x, data_y = [], []

        if output_stats:
            print(f"({title}) Loaded {len(files)} files with cumulatively {len(points)} points (skipped {skipped})")
            print(f"Mean (only accepted): {np.exp(np.mean(points, axis=0))}")

        sns.kdeplot(ax=ax, x=data_x, y=data_y, fill=True, cmap=cmap, bw_adjust=smooth, thresh=thresh, levels=levels)
        ax.scatter(*np.log(Theta_grid), s=40, marker="x", color="black")

        # enable to also plot points
        ax.scatter(data_x, data_y, color="black", s=1)

        # ax.scatter(*np.random.normal(loc=-3, scale=0.2, size=(2,60)))

        ax.axline((0, 0), (1, 1), linestyle="--", linewidth=1, color="black")

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))

        ax.set_xticks(np.log(np.logspace(-3, -1, 3)))
        ax.set_xticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)
        ax.set_yticks(np.log(np.logspace(-3, -1, 3)))
        ax.set_yticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)

        ax.grid(which="minor", ls="--", linewidth=0.3)

        ax.set_xlim(np.log(xlim))
        ax.set_ylim(np.log(ylim))

        ax.set_aspect("equal")

        ax.text(np.log(xlim[1]) - 0.1, np.log(ylim[0]) + 0.1, title, size=20, ha="right", va="bottom")

    fig.text(0.5, 0.02, r"recovery rate $\alpha$ [day$^{-1}$]", size=25, ha="center")
    fig.text(0.01, 0.5, r"infection rate $\beta$ [day$^{-1}$]", size=25, va="center", rotation="vertical")

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(plot_path + ".png")
    fig.savefig(plot_path + ".pdf")


@app.command()
def trace(month: int = typer.Option(...),
          run: int = typer.Option(...)):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(10, 5), dpi=300)

    # HMC
    hmc_path = join(HMC_POINTS_DIR, f"{month:02d}", f"hmc-points-{month:02d}-{run:02d}.csv")
    with open(hmc_path, "r") as f:
        lines = f.readlines()

    alphas, betas = zip(*[[float(v) for v in l.strip().split(",")] for l in lines])
    axes[0][0].plot(alphas)
    axes[1][0].plot(betas)

    # MH
    mh_path = join(MH_POINTS_DIR, f"{month:02d}", f"mh-points-{month:02d}-{run:02d}.csv")
    with open(mh_path, "r") as f:
        lines = f.readlines()

    alphas, betas = zip(*[[float(v) for v in l.strip().split(",")] for l in lines])
    axes[0][1].plot(alphas)
    axes[1][1].plot(betas)

    # headers
    axes[0][0].set_title("HMC")
    axes[0][1].set_title("MH")
    axes[0][0].set_ylabel(r"$\alpha$", rotation=0)
    axes[1][0].set_ylabel(r"$\beta$", rotation=0)

    # limits
    for row in axes:
        for col in row:
            col.set_xlim(0, len(alphas))

    fig.tight_layout()

    fig.savefig(join(PLOTS_DIR, "trace.png"))
    fig.savefig(join(PLOTS_DIR, "trace.pdf"))


def diagnostics_plot(sampling, metric, show=False):
    if sampling == "hmc":
        points_dir = HMC_POINTS_DIR
        ess_path = HMC_ESS_PATH

    elif sampling == "mh":
        points_dir = MH_POINTS_DIR
        ess_path = MH_ESS_PATH

    if metric == "ess":
        func = partial(arviz.ess, method="folded")
    elif metric == "rhat":
        func = partial(arviz.rhat, method="folded")

    fig, axes = plt.subplots(6, 2, figsize=(5, 15), dpi=300)

    for m, axs in enumerate(axes, start=3):
        month_dir = join(points_dir, f"{m:02d}")

        os.makedirs(month_dir, exist_ok=True)

        all_alphas = []
        all_betas = []

        for c, file in enumerate(sorted(os.listdir(month_dir))):
            with open(join(month_dir, file), "r") as f:
                lines = f.readlines()

            chain_alphas, chain_betas = zip(*[[float(v) for v in l.strip().split(",")] for l in lines])
            all_alphas.append(chain_alphas)
            all_betas.append(chain_betas)

        if len(all_alphas) == 0:
            continue

        min_size = min(len(l) for l in all_alphas + all_betas)
        all_alphas = [l[:min_size] for l in all_alphas]
        all_betas = [l[:min_size] for l in all_betas]

        all_alphas = np.array(all_alphas)
        all_betas = np.array(all_betas)

        es_alpha = []
        es_beta = []
        for t in range(5, all_alphas.shape[1]):
            es_alpha.append(func(all_alphas[:, :t]))
            es_beta.append(func(all_betas[:, :t]))

        ax1, ax2 = axs

        ax1.plot(es_alpha)
        ax2.plot(es_beta)

        for ax in axs:
            if metric == "ess":
                ax.set_ylim(0, all_alphas.size)
            elif metric == "rhat":
                ax.set_ylim(1, 1.5)
                ax.axhline(1.1, color="black", linestyle="--")

        ax1.set_ylabel(str(m))

    axes[0][0].set_title("alpha")
    axes[0][1].set_title("beta")

    fig.suptitle(f"{sampling} {metric}")
    fig.tight_layout()

    fig.savefig(ess_path + ".png")
    if show:
        fig.show()


def timeline_plot(show=False):
    rcParams["axes.labelsize"] = 15
    rcParams["xtick.labelsize"] = 15

    data_sir = data.covid19_austria_daily("2020-03-01", "2020-09-01", cache_path=AUSTRIA_DATA_CACHE_PATH)

    df = pd.DataFrame(data_sir, columns=["t", "S", "I", "R"])
    df["t"] = df["t"].apply(lambda t: dt.datetime(2020, 3, 1) + dt.timedelta(days=t))
    df = df.set_index("t")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    df["I"].plot(ax=ax, color="lightcoral")
    ax.set_ylabel("infected", color="lightcoral")
    ax.set_ylim(0, 9000)
    ax.set_yticks(np.linspace(0, 10000, 6))

    ax2 = ax.twinx()

    df["R"].plot(ax=ax2, color="black")
    ax2.set_ylabel("recovered")
    ax2.set_ylim(0, 25000)
    ax2.set_yticks(np.linspace(0, 25000, 6))
    ax2.grid(False)

    ax.set_xlabel("date in 2020")

    ax.xaxis.set_major_locator(dates.MonthLocator())
    # 16 is a slight approximation since months differ in number of days.
    ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=16))

    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(dates.DateFormatter("%B"))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment("center")

    fig.tight_layout()

    fig.savefig(AUSTRIA_TIMELINE_PLOT_PATH + ".png")
    fig.savefig(AUSTRIA_TIMELINE_PLOT_PATH + ".pdf")
    if show:
        fig.show()


if __name__ == '__main__':
    app()
