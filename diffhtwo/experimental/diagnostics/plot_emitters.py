import matplotlib.pyplot as plt

from ..data_loaders.load_minerva import (
    MinervaFilters,
    get_filt_indx,
    pick_line_emitters,
)
from .plot_contour import parse_color_labels

plt.rc("font", family="serif", serif=["Times New Roman"])
plt.rc(
    "mathtext",
    fontset="custom",
    rm="Times New Roman",
    it="Times New Roman:italic",
    bf="Times New Roman:bold",
)


def plot_line_emitters_minerva(
    ran_key,
    minerva,
    space_idx,
    savedir,
    Sigma_thresh=3.0,
    figsize=(7.1, 3.8),
    xlim=(20, 28),
    ylim=(-1, 2),
    plt_show=True,
):
    labelsize = 9
    fig, ax = plt.subplots(
        2, 4, figsize=figsize, constrained_layout=True, squeeze=False
    )
    fig.get_layout_engine().set(
        h_pad=0.0, wspace=0.05, hspace=0.05, rect=(0, 0, 1, 0.925)
    )
    row = 0
    col = 0
    for idx in space_idx:
        space = minerva.spaces[idx]._fields[-1]
        print(space)
        xlabel, ylabel = parse_color_labels(space)
        z_min, z_max = minerva.spaces[idx].z_min, minerva.spaces[idx].z_max
        mag_idx, b, c = get_filt_indx(space, MinervaFilters)

        MB = minerva.mags[:, mag_idx]
        BB = minerva.mags[:, b]

        MB_err = minerva.mags_err[:, mag_idx]
        BB_err = minerva.mags_err[:, b]

        sel = minerva.sels[:, mag_idx] & minerva.sels[:, b]
        z_sel = (minerva.redshift > z_min) & (minerva.redshift < z_max)
        emitter_sel = pick_line_emitters(
            BB, MB, BB_err, MB_err, Sigma_thresh=Sigma_thresh
        )

        excess_color = BB - MB
        ax[row][col].scatter(
            MB[sel & z_sel],
            excess_color[sel & z_sel],
            s=0.2,
            alpha=0.8,
            c="gray",
            label="all",
        )
        ax[row][col].scatter(
            MB[sel & z_sel & emitter_sel],
            excess_color[sel & z_sel & emitter_sel],
            s=10,
            alpha=0.5,
            marker="s",
            facecolor="none",
            edgecolor="purple",
            linewidths=0.5,
            label="emitters",
        )
        ax[row][col].set_xlim(xlim)
        ax[row][col].set_ylim(ylim)
        ax[row][col].set_xlabel(xlabel)
        ax[row][col].set_ylabel(ylabel)
        ax[row][col].minorticks_on()
        ax[row][col].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[row][col].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        col += 1
        if col == 4:
            row += 1
            col = 0

    handles, labels = ax[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        fontsize=labelsize,
        frameon=False,
    )
    leg.legend_handles[0].set_sizes([20])
    leg.legend_handles[1].set_sizes([40])

    fig.savefig(f"{savedir}/minerva_line_emitters.png", dpi=600)
    if plt_show:
        plt.show()
    plt.close()
