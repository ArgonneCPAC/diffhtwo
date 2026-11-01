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
    z_data = data.spaces[0]
    z_data_model = add_N_model_to_spaces(ran_key, param_collection, z_data, mag_thresh)
    fields = [f for f in z_data_model._fields if f not in {"z_min", "z_max", "lc_data"}]

    labelsize = 10
    fontsize = 10
    for f in fields:
        space = getattr(z_data_model, f)
        if isinstance(space, list):
            fig, ax = plt.subplots(
                1, len(space), figsize=(7.1, 2.5), constrained_layout=True
            )
            fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))

            i = 0
            for cond_space in space:
                cond_band, color = parse_color_labels(type(cond_space).__name__)
                bin_lo = cond_space.bin_lo.ravel()
                bin_hi = cond_space.bin_hi.ravel()
                bin_centers = (bin_lo + bin_hi) / 2

                edges = np.append(bin_lo, bin_hi[-1])

                n_data = cond_space.N_data / cond_space.vol_mpc3_data
                n_model = cond_space.N_model / cond_space.vol_mpc3_model

                ax[i].stairs(pdf(cond_space.N_data, edges), edges, label="data")
                ax[i].stairs(pdf(cond_space.N_model, edges), edges, label="model")

                # ax[i].scatter(bin_centers, np.log10(n_data), s=1)
                # ax[i].plot(bin_centers, np.log10(n_model))

                ax[i].set_title(
                    f"{np.round(cond_space.cond_lo, 2)} < {cond_band} < {np.round(cond_space.cond_hi, 2)}"
                )
                ax[i].set_xlabel(color)

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

                ax[i].set_xlim(0, 1.5)
                i += 1
            ax[0].set_ylabel("PDF", fontsize=fontsize)
            ax[0].legend()

            fig.savefig(savedir + "/" + run_label + "_app_mag_funcs.png", dpi=400)
            if plt_show:
                plt.show()
            plt.close()
