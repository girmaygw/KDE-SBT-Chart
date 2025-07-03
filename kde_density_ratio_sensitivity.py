#KDE plot comparing sensitive vs. non-sensitive soils using density ratio, Heat map
############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D

# Load & filter data
def load_and_filter_data(path, sheet, soil_types_column):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(subset=["Qt", "Bq", "Soil type"])
    df = df[df["Soil type"].isin(soil_types_column)]
    df = df[(df["Qt"] > 0) & (df["Bq"] > -0.5)]
    df["logQt"] = np.log10(df["Qt"])
    return df

# KDE + R-heatmap + R-contours + Marginals
def plot_kde_with_sensitivity(df, x="Bq", y="logQt", hue="Soil type"):
    classes = list(df[hue].unique())
    colors = {s: plt.get_cmap("tab10").colors[i] for i, s in enumerate(classes)}

    # Grid layout
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
    ax_main  = fig.add_subplot(gs[1:, :-1])
    ax_top   = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    ax_main.set_xlim(-0.5, 1.7)
    ax_main.set_ylim(0, 3)
    ax_main.set_xlabel(r"$B_q$")
    ax_main.set_ylabel(r"$Q_t$")
    ax_main.set_yticks(np.linspace(0, 3, 4))
    ax_main.set_yticklabels(["1", "10", "100", "1000"])
    ax_main.set_title("KDE + Continuous Sensitivity Index (R)")

    # evaluation grid
    N = 1000
    gx = np.linspace(-0.5, 1.7, N)
    gy = np.linspace(0, 3, N)
    X, Y = np.meshgrid(gx, gy)
    pos = np.vstack([X.ravel(), Y.ravel()])

    # compute class KDEs
    densities = np.zeros((len(classes), N, N))
    for i, s in enumerate(classes):
        pts = df[df[hue] == s][[x, y]].values.T
        if pts.shape[1] >= 2:
            kde = gaussian_kde(pts, bw_method="scott")
            densities[i] = kde(pos).reshape(N, N)

    # mask noise below 5% of peak
    for i in range(len(classes)):
        peak = densities[i].max()
        densities[i][densities[i] < 0.05 * peak] = 0

    if len(classes) == 2:
        f0 = densities[0]
        f1 = densities[1]
        R = f1 / (f0 + f1 + 1e-12)

        # Custom colormap with white background for R=0
        cmap_r = plt.get_cmap("RdYlBu")
        new_cmap = LinearSegmentedColormap.from_list("R_white_bg", ["white"] + [cmap_r(i) for i in range(cmap_r.N)], N=256)

        im = ax_main.imshow(
            R,
            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
            origin="lower",
            aspect="auto",
            cmap=new_cmap,
            vmin=0, vmax=1,
            zorder=1
        )
        cbar = fig.colorbar(im, ax=ax_main, pad=0.02)
        cbar.set_label("Sensitivity index R")

        levels = np.linspace(0.1, 0.9, 9)
        cs = ax_main.contour(
            X, Y, R,
            levels=levels,
            colors="k",
            linewidths=0.8,
            linestyles="--",
            zorder=2
        )
        fmt = {lev: f"{int(lev*100)}%" for lev in levels}
        ax_main.clabel(cs, fmt=fmt, inline=True, fontsize=8)

    # Marginal KDEs
    for i, s in enumerate(classes):
        sub = df[df[hue] == s]
        if len(sub) < 2:
            continue
        kde_x = gaussian_kde(sub[x])
        ax_top.plot(gx, kde_x(gx), color=colors[s])
        kde_y = gaussian_kde(sub[y])
        ax_right.plot(kde_y(gy), gy, color=colors[s])

    ax_top.axis("off")
    ax_right.axis("off")

    # Legend
    contour_line = Line2D([0], [0], color="black", linestyle="--", label="R-contours")
    ax_main.legend(handles=[contour_line], title="Sensitivity Index (R)", loc="upper right")

    plt.tight_layout()
    plt.show()

