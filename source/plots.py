import datetime as dt
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, dates, ticker
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import data
import tensir
from paths import AUSTRIA_MONTHLY_HMC_POINTS_DIR, PLOTS_DIR, AUSTRIA_MONTHLY_HMC_PLOT_PATH, AUSTRIA_DATA_CACHE_PATH, \
    AUSTRIA_TIMELINE_PLOT_PATH

rcParams["text.usetex"] = True
sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "axes.edgecolor": "0.15",
                            "font.family": "serif", "font.serif": ["Computer Modern"]})


def density_plots(show=False, output_stats=False):
    levels = 5
    burn_in = 10  # number of points of each run to skip
    xlim = (0.01, 0.3)
    ylim = (0.01, 0.3)
    smooth = 3.5
    thresh = 0.02

    data_files = []
    for m in range(3, 9):
        month_dir = join(AUSTRIA_MONTHLY_HMC_POINTS_DIR, f"{m:02d}")
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

        sns.kdeplot(ax=ax, x=data_x, y=data_y, shade=True, cmap=cmap, bw_adjust=smooth, thresh=thresh, levels=levels)
        ax.scatter(*np.log(Theta_grid), s=40, marker="x", color="black")

        # ax.scatter(data_x, data_y, color="black", s=1)

        ax.axline((0, 0), (1, 1), linestyle="--", linewidth=1, color="black")

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))

        ax.set_xticks(np.log(np.logspace(-3, -1, 3)))
        ax.set_xticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)
        ax.set_yticks(np.log(np.logspace(-3, -1, 3)))
        ax.set_yticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)

        ax.grid(b=True, which="minor", ls="--", linewidth=0.3)

        ax.set_xlim(np.log(xlim))
        ax.set_ylim(np.log(ylim))

        ax.set_aspect("equal")

        ax.text(np.log(xlim[1]) - 0.1, np.log(ylim[0]) + 0.1, title, size=20, ha="right", va="bottom")

    fig.text(0.5, 0.02, r"recovery rate $\alpha$ [day$^{-1}$]", size=25, ha="center")
    fig.text(0.01, 0.5, r"infection rate $\beta$ [day$^{-1}$]", size=25, va="center", rotation="vertical")

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(AUSTRIA_MONTHLY_HMC_PLOT_PATH + ".png")
    fig.savefig(AUSTRIA_MONTHLY_HMC_PLOT_PATH + ".pdf")
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


def main():
    # density_plots(show=True, output_stats=True)
    trace_plot(show=True)
    # psrf_plot(show=True)
    # timeline_plot(show=True)


if __name__ == '__main__':
    main()
