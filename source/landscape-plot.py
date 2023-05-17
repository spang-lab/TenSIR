import os
from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

from paths import PROJECT_ROOT, PLOTS_DIR

landscape_points_dir = join(PROJECT_ROOT, "landscape")


def plot(mode):
    xlim = (0.001, 1.)
    ylim = (0.001, 1.)

    data_files = []
    for m in range(3, 9):
        data_files.append([join(landscape_points_dir, f) for f in os.listdir(landscape_points_dir)
                           if f.startswith(f"0{m}")])

    titles = ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020", "August 2020"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)

    for i, (files, title) in enumerate(zip(data_files, titles)):

        ax = axes[i // 3, i % 3]

        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file))

        if len(dfs) == 0:
            continue

        df = pd.concat(dfs)
        df = df.sort_values(by=["alpha", "beta"]).reset_index(drop=True)

        df[["alpha", "beta"]] = np.log(df[["alpha", "beta"]])
        df["grad"] = -np.sqrt(df["d_alpha"] ** 2 + df["d_beta"] ** 2)

        x = df["alpha"].values
        y = df["beta"].values
        z = df[mode].values

        finite = np.isfinite(z)
        vmin = np.min(z[finite]) - 1
        z[~finite] = vmin * 2

        cmap = mpl.colormaps.get_cmap("viridis")
        cmap.set_under("lightgrey")
        ax.tricontourf(x, y, z, levels=30, cmap=cmap, vmin=vmin)

        if mode == "grad":
            s = 0.0005
            for _, (alpha, beta, d_alpha, d_beta) in df[["alpha", "beta", "d_alpha", "d_beta"]].iterrows():
                if np.isfinite((d_alpha, d_beta)).all():
                    ax.arrow(alpha, beta, s * d_alpha, s * d_beta, head_width=0.02)

        ax.axline((0, 0), (1, 1), linestyle="--", linewidth=1, color="black")

        verts = np.log([(0.01, 0.01), (0.3, 0.01), (0.3, 0.3), (0.01, 0.3)])
        ax.add_patch(Polygon(verts, facecolor="none", edgecolor="black", linewidth=2, linestyle="--"))

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.3f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.exp(x):.3f}"))

        ax.set_xticks(np.log(np.logspace(-3, 0, 4)))
        ax.set_xticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)
        ax.set_yticks(np.log(np.logspace(-3, 0, 4)))
        ax.set_yticks(np.log([np.linspace(1, 9, 9) * 10 ** i for i in range(-3, 0)]).flatten(), minor=True)

        ax.grid(which="minor", ls="--", linewidth=0.3)

        ax.set_xlim(np.log(xlim))
        ax.set_ylim(np.log(ylim))

        ax.set_aspect("equal")

        ax.text(np.log(xlim[1]) - 0.1, np.log(ylim[0]) + 0.1, title, size=20, ha="right", va="bottom")

    fig.text(0.5, 0.02, r"recovery rate $\alpha$ [day$^{-1}$]", size=25, ha="center")
    fig.text(0.01, 0.5, r"infection rate $\beta$ [day$^{-1}$]", size=25, va="center", rotation="vertical")

    if mode == "ll":
        fig.suptitle("Log-likelihood landscape")
    if mode == "grad":
        fig.suptitle("Gradient landscape")

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(join(PLOTS_DIR, f"landscape-{mode}.png"))


if __name__ == "__main__":
    plot("ll")
    plot("grad")
