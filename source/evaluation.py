import datetime as dt
import os
from os.path import join

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib import rcParams, dates, ticker, cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import tensir
from paths import RESULTS_DIR, PLOTS_DIR, AUSTRIA_DATA_CACHE_PATH

rcParams["text.usetex"] = True
sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "axes.edgecolor": "0.15",
                            "font.family": "serif", "font.serif": ["Computer Modern"]})

SAMPLINGS = ["nuts", "mh"]
MONTHS = [(3, "March"), (4, "April"), (5, "May"), (6, "June"), (7, "July"), (8, "August")]

app = typer.Typer(no_args_is_help=True, add_completion=False)


def load_data():
    print("Loading data...")
    all_data = []
    for sampling in SAMPLINGS:
        points_dir = join(RESULTS_DIR, f"{sampling}-points")

        month_datas = []
        for m, month in MONTHS:
            files = [f for f in os.listdir(points_dir) if f.startswith(f"{sampling}-points-0{m}")]
            datas = [arviz.from_netcdf(join(points_dir, f)) for f in files]

            data = arviz.concat(*datas, dim="chain")
            month_datas.append(data)

        all_data.append(month_datas)

    return all_data


@app.command(help="Print diagnostic tables in LaTeX format")
def tables():
    data = load_data()

    for sampling, month_datas in zip(SAMPLINGS, data):

        print(f"\n{sampling.upper()}:\n")

        print(r"\begin{tabular}{c|c|c|c}")
        print(r"month & runtime & ESS $\alpha$ & ESS $\beta$ \\")
        print(r"\hline")

        for (_, month), data in zip(MONTHS, month_datas):
            runtime = np.mean(data.sample_stats.combined_sampling_time)
            runtime_order = int(np.log10(runtime))

            ess_alpha, ess_beta = arviz.ess(data, method="folded").theta.values

            print(f"{month} & ${runtime / 10 ** runtime_order:.1f} \\cdot 10^{runtime_order}$ & "
                  f"${ess_alpha:.0f}$ & ${ess_beta:.0f}$ \\\\")

        print(r"\end{tabular}")


@app.command(help="Create a trace plot in the ./results/plots directory.")
def trace_plot(month: int = typer.Option(..., help="Select a month (3-8)"),
               chain: int = typer.Option(..., help="Select a chain (0-9)")):
    data = load_data()

    fig, axeses = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 5), dpi=300)

    for i, axes in enumerate(axeses):

        for j, ax in enumerate(axes):
            warmup = data[j][month - 3].warmup_posterior.theta.values[chain, :, i]
            samples = data[j][month - 3].posterior.theta.values[chain, :, i]

            w = len(warmup)
            ax.plot(range(w), warmup, color="grey")
            ax.plot(range(w, w + len(samples)), samples, color="tab:blue")

    # headers
    axeses[0][0].set_title("HMC")
    axeses[0][1].set_title("MH")
    axeses[0][0].set_ylabel(r"$\log\alpha$", rotation=90, size=16)
    axeses[1][0].set_ylabel(r"$\log\beta$", rotation=90, size=16)

    # fig.suptitle(f"Trace plot month {month} chain {chain}")

    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig.savefig(join(PLOTS_DIR, f"trace-{month:02d}-{chain:02d}.png"))
    fig.savefig(join(PLOTS_DIR, f"trace-{month:02d}-{chain:02d}.pdf"))


@app.command(help="Create the timeline plot in the ./results/plots directory.")
def timeline_plot():
    rcParams["axes.labelsize"] = 15
    rcParams["xtick.labelsize"] = 15

    data_sir = tensir.data.covid19_austria_daily("2020-03-01", "2020-09-01", cache_path=AUSTRIA_DATA_CACHE_PATH)

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


@app.command(help="Create the density plot in the ./results/plots directory.")
def density_plot(scatter: bool = typer.Option(False, help="Add the underlying points as a scatter plot")):
    data = load_data()

    vals = [arviz.extract(m.posterior).theta.values for m in data[0]]

    levels = 5
    xlim = (0.02, 0.3)
    ylim = (0.02, 0.3)
    smooth = 7
    thresh = 0.001

    titles = ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020", "August 2020"]

    grid_searched = [tensir.optimize.grid.grid_search_deterministic(
        tensir.data.covid19_austria_daily(f"2020-0{m}-01", f"2020-0{m + 1}-01", cache_path=AUSTRIA_DATA_CACHE_PATH),
        (xlim[0], ylim[0]), (xlim[1], ylim[1]), 40)
        for m in range(3, 9)]

    print(grid_searched)

    greys = cm.get_cmap("Greys", 100)
    cmap = ListedColormap(greys(np.linspace(0.2, 0.8)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)

    for i, (points, title, Theta_grid) in enumerate(zip(vals, titles, grid_searched)):

        ax = axes[i // 3, i % 3]

        data_x, data_y = points
        print(f"{title}: {len(data_x)} points")

        sns.kdeplot(ax=ax, x=data_x, y=data_y, fill=True, cmap=cmap, bw_adjust=smooth, thresh=thresh, levels=levels)
        ax.scatter(*np.log(Theta_grid), s=40, marker="x", color="black")

        # enable to also plot points
        if scatter:
            ax.scatter(data_x, data_y, color="black", s=1)

        ax.axline((0, 0), (1, 1), linestyle="--", linewidth=1, color="black")

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.2f}"))

        ax.set_xticks(np.log([0.02, 0.1, 0.3]))
        ax.set_xticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)
        ax.set_yticks(np.log([0.02, 0.1, 0.3]))
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

    fig.savefig(join(PLOTS_DIR, f"density.png"))
    fig.savefig(join(PLOTS_DIR, f"density.pdf"))


if __name__ == "__main__":
    app()
