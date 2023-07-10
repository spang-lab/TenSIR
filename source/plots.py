import datetime as dt
import os
from enum import Enum
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
from paths import PLOTS_DIR, AUSTRIA_DATA_CACHE_PATH, HMC_POINTS_DIR, MH_POINTS_DIR
from tensir import data

rcParams["text.usetex"] = True
sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "axes.edgecolor": "0.15",
                            "font.family": "serif", "font.serif": ["Computer Modern"]})


class Sampling(str, Enum):
    HMC = "hmc"
    MH = "mh"


app = typer.Typer()


@app.command(help="Create a density plot(s).")
def density(sampling: Sampling = typer.Option(..., help="Specify the results of which sampler to use."),
            scatter: bool = typer.Option(False, help="Whether to scatter the points themselves on the KDE plot."),
            output_stats: bool = typer.Option(False, help="Whether to output statistics to the terminal.")):
    levels = 5
    burn_in = 200  # number of points of each run to skip
    xlim = (0.01, 0.3)
    ylim = (0.01, 0.3)
    smooth = 5
    thresh = 0.002

    if sampling == "hmc":
        points_dir = HMC_POINTS_DIR

    elif sampling == "mh":
        points_dir = MH_POINTS_DIR

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
        if scatter:
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

    fig.savefig(join(PLOTS_DIR, f"density-{sampling}.png"))
    fig.savefig(join(PLOTS_DIR, f"density-{sampling}.pdf"))


@app.command(help="Create a trace plot(s).")
def trace(month: int = typer.Option(..., help="Select a month (3-8)"),
          run: int = typer.Option(..., help="Select a run")):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(10, 5), dpi=300)

    burnin = 200

    # HMC
    hmc_path = join(HMC_POINTS_DIR, f"{month:02d}", f"hmc-points-{month:02d}-{run:02d}.csv")
    with open(hmc_path, "r") as f:
        lines = f.readlines()

    alphas, betas = zip(*[[float(v) for v in l.strip().split(",")] for l in lines])
    axes[0][0].plot(range(0, burnin), alphas[:burnin], color="grey")
    axes[0][0].plot(range(burnin, 1000), alphas[burnin:], color="tab:blue")
    axes[1][0].plot(range(0, burnin), betas[:burnin], color="grey")
    axes[1][0].plot(range(burnin, 1000), betas[burnin:], color="tab:blue")

    # MH
    mh_path = join(MH_POINTS_DIR, f"{month:02d}", f"mh-points-{month:02d}-{run:02d}.csv")
    with open(mh_path, "r") as f:
        lines = f.readlines()

    alphas, betas = zip(*[[float(v) for v in l.strip().split(",")] for l in lines])
    axes[0][1].plot(range(0, burnin), alphas[:burnin], color="grey")
    axes[0][1].plot(range(burnin, 1000), alphas[burnin:], color="tab:blue")
    axes[1][1].plot(range(0, burnin), betas[:burnin], color="grey")
    axes[1][1].plot(range(burnin, 1000), betas[burnin:], color="tab:blue")

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

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(join(PLOTS_DIR, f"trace-{month:02d}-{run:02d}.png"))
    fig.savefig(join(PLOTS_DIR, f"trace-{month:02d}-{run:02d}.pdf"))


@app.command(help="Create an autocorrelation plot(s).")
def autocorrelation(month: int = typer.Option(..., help="Select a month (3-8)")):
    burnin = 200

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(10, 5), dpi=300)

    for param, row in enumerate(axes):

        for ax, sampling in zip(row, ("hmc", "mh")):

            chains = []
            for run in range(10):
                with open(f"results/{sampling}-points/{month:02d}/{sampling}-points-{month:02d}-{run:02d}.csv",
                          "r") as f:
                    lines = f.readlines()

                lines = [l.strip().split(",") for l in lines]
                pairs = [(float(a), float(b)) for a, b in lines]

                chains.append([p[param] for p in pairs])

            data = np.array(chains)[:, burnin:]

            arviz.plot_autocorr(data, max_lag=50, ax=ax)

            ax.set_title("")

    # headers
    axes[0][0].set_title("HMC")
    axes[0][1].set_title("MH")
    axes[0][0].set_ylabel(r"$\alpha$", rotation=0)
    axes[1][0].set_ylabel(r"$\beta$", rotation=0)

    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(join(PLOTS_DIR, f"autocorrelation-{month:02d}.png"))
    fig.savefig(join(PLOTS_DIR, f"autocorrelation-{month:02d}.pdf"))


@app.command(help="Create the timeline plot(s).")
def timeline():
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

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(join(PLOTS_DIR, f"timeline.png"))
    fig.savefig(join(PLOTS_DIR, f"timeline.pdf"))


@app.command(help="Create all plot(s).")
def all():
    timeline()

    for month in range(3, 9):
        autocorrelation(month)
        trace(month, run=0)

    for sampling in (Sampling.HMC, Sampling.MH):
        density(sampling)


if __name__ == '__main__':
    app()
