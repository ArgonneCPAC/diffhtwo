import numpy as np
from matplotlib import pyplot as plt

from ..kernels.N_phot import add_N_model_to_spaces
from .plot_contour import parse_color_labels

plt.rc("font", family="serif", serif=["Times New Roman"])


def pdf(counts, edges):
    return counts / (counts.sum() * np.diff(edges))


def plot_cond_color(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    run_label,
    savedir,
    plt_show=True,
):
    labelsize = 9
    legendsize = 9
    fontsize = 10

    n_z_bins = len(data.spaces)
    for z in range(n_z_bins):
        z_data = data.spaces[z]
        z_data_model = add_N_model_to_spaces(
            ran_key, param_collection, z_data, mag_thresh
        )
        fields = [
            f for f in z_data_model._fields if f not in {"z_min", "z_max", "lc_data"}
        ]

        for sp in fields:
            if sp.startswith("condc_"):
                space = getattr(z_data_model, sp)
                fig, ax = plt.subplots(
                    1,
                    len(space),
                    figsize=(7.1, min(max(7.1 / len(space), 1.8), 3.2)),
                    constrained_layout=True,
                    sharex=True,
                    squeeze=False,
                )
                fig.get_layout_engine().set(rect=(0, 0, 1, 0.84))
                ax = ax[0]

                zlo = space[0].zlo
                zhi = space[0].zhi
                fig.suptitle(
                    f"{float(zlo):.2f} < z < {float(zhi):.2f}",
                    fontsize=fontsize,
                    y=0.99,
                )

                xlo = min(float(s.bin_lo.min()) for s in space)
                xhi = max(float(s.bin_hi.max()) for s in space)

                for i, cond_space in enumerate(space):
                    cond_band, color = parse_color_labels(type(cond_space).__name__)
                    bin_lo = cond_space.bin_lo.ravel()
                    bin_hi = cond_space.bin_hi.ravel()
                    # bin_centers = (bin_lo + bin_hi) / 2

                    edges = np.append(bin_lo, bin_hi[-1])

                    # n_data = cond_space.N_data / cond_space.vol_mpc3_data
                    # n_model = cond_space.N_model / cond_space.vol_mpc3_model

                    ax[i].stairs(pdf(cond_space.N_data, edges), edges, label="data")
                    ax[i].stairs(pdf(cond_space.N_model, edges), edges, label="diffsky")

                    # ax[i].scatter(bin_centers, np.log10(n_data), s=1)
                    # ax[i].plot(bin_centers, np.log10(n_model))

                    ax[i].set_title(
                        f"{float(cond_space.cond_lo):.1f} < {cond_band} < {float(cond_space.cond_hi):.1f}",
                        fontsize=labelsize,
                    )
                    ax[i].set_xlabel(color, fontsize=labelsize)

                    ax[i].minorticks_on()
                    ax[i].tick_params(
                        which="major",
                        direction="in",
                        top=True,
                        right=True,
                        length=6,
                        width=1,
                        labelsize=labelsize,
                    )
                    ax[i].tick_params(
                        which="minor",
                        direction="in",
                        top=True,
                        right=True,
                        length=3,
                        width=0.8,
                        labelsize=labelsize,
                    )

                    ax[i].set_xlim(xlo, xhi)

                ax[0].set_ylabel("PDF", fontsize=labelsize)
                handles, labels = ax[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.96),
                    ncol=len(labels),
                    frameon=False,
                    fontsize=legendsize,
                )

                fig.savefig(
                    f"{savedir}/{run_label}_z{float(zlo):.2f}-{float(zhi):.2f}_{sp}.png",
                    dpi=400,
                )
                if plt_show:
                    plt.show()
                plt.close()
