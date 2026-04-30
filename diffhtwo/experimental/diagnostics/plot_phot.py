import jax.numpy as jnp
import numpy as np
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data
from ..n_specphot import get_colors_mags, mag_kern

blue = "#1E90FF"  # DodgerBlue
orange = "#FF8C00"  # DarkOrange
# blue = "#4169E1"  # RoyalBlue
# orange = "#D2691E"  # Chocolate
# blue = "#00BFFF"  # DeepSkyBlue
# orange = "#FFA500"  # Orange

color1 = orange
color2 = "k"
color_data = blue

alpha1 = 1.0
alpha2 = 0.7
alpha_data = 0.5


lw = 1.5
fontsize = 40
labelsize = 40
legend_fontsize = 30


try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


mpl.rcParams["axes.linewidth"] = 2.5


def plot_n_colors_mag(
    dataset,
    data_label,
    param_collection1,
    label1,
    dimension_labels,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    suptitle,
    savedir,
    param_collection2=None,
    label2=None,
    lg_n_thresh=None,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=10000,
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    dataset_colors_mag = dataset.dataset
    data_sky_area_degsq = dataset.data_sky_area_degsq

    z_min, z_max = np.round(z_min, 2), np.round(z_max, 2)
    z_mask = (dataset_colors_mag[:, -1] > z_min) & (dataset_colors_mag[:, -1] < z_max)
    dataset_colors_mag_z = dataset_colors_mag[z_mask]
    data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=z_min, zhigh=z_max).value

    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        dataset.tcurves,
        z_phot_table,
    )
    obs_color_mag1, weights1 = get_colors_mags(
        ran_key,
        param_collection1,
        lc_data,
        dataset.mag_columns,
        dataset.mag_thresh_column,
        dataset.mag_thresh,
        dataset.frac_cat,
    )

    n_panels = obs_color_mag1.shape[1]

    if data_label == "SDSS":
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if data_label == "FENIKS":
        fig_width = 2.25 * n_panels
        fig_height = n_panels / 1.5

        fontsize = 2.25 * n_panels
        labelsize = 1.75 * n_panels
        legend_fontsize = 1.25 * n_panels

    fig, ax = plt.subplots(
        2,
        n_panels,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.subplots_adjust(
        left=0.05, hspace=0, top=0.875, right=0.99, bottom=0.15, wspace=0.0
    )
    fig.suptitle(
        suptitle + "   |   " + str(z_min) + " < z < " + str(z_max), fontsize=24
    )
    for i in range(0, n_panels):
        if i == n_panels - 1:
            bins = np.linspace(
                dataset_colors_mag_z[:, i].min(),
                dataset_colors_mag_z[:, i].max(),
                20,
            )
        else:
            std = np.std(dataset_colors_mag_z[:, i])
            med = np.median(dataset_colors_mag_z[:, i])
            bins = np.linspace(
                med - (5 * std),
                med + (6 * std),
                20,
            )

        bin_centers = (bins[1:] + bins[:-1]) / 2
        ax[0, i].set_xlim(bins[0], bins[-1])
        ax[1, i].set_xlim(bins[0], bins[-1])

        n_data, bin_edges, _ = ax[0, i].hist(
            dataset_colors_mag_z[:, i],
            weights=np.ones_like(dataset_colors_mag_z[:, i]) * (1 / data_vol_mpc3),
            bins=bins,
            color="k",
            label=data_label,
            alpha=0.5,
        )

        n_diffsky, _, _ = ax[0, i].hist(
            obs_color_mag1[:, i],
            weights=weights1 * (1 / lc_data.lc_tot_vol_mpc3),
            bins=bins,
            color="deepskyblue",
            label=label1,
            alpha=0.5,
        )

        if i == n_panels - 1:
            ylim_top = 2 * n_diffsky.max()

        ax[0, i].set_yscale("log")
        ax[0, i].tick_params(axis="both", direction="in", labelsize=labelsize)

        offset = n_data / n_diffsky
        ax[1, i].plot(bin_centers, offset, lw=2.0, color="k")
        ax[1, i].set_ylim(0.09, 10.1)
        ax[1, i].set_yscale("log")
        ax[1, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
        ax[1, i].tick_params(axis="both", direction="in", labelsize=labelsize)
        ax_offset_yticks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
        ax[1, i].set_yticks(ax_offset_yticks)
        ax[1, i].set_yticklabels(["", "0.2", "0.5", "1", "2", "5", ""])
        ax[1, i].axhspan(
            ax_offset_yticks[2], ax_offset_yticks[4], color="orange", alpha=0.25
        )
        ax[1, i].axhspan(
            ax_offset_yticks[1], ax_offset_yticks[2], color="orange", alpha=0.5
        )
        ax[1, i].axhspan(
            ax_offset_yticks[4], ax_offset_yticks[5], color="orange", alpha=0.5
        )
        ax[1, i].axhspan(0, ax_offset_yticks[1], color="orange", alpha=0.8)
        ax[1, i].axhspan(ax_offset_yticks[5], 10, color="orange", alpha=0.8)
        ax[1, i].axhline(1, color="green", alpha=0.5, lw=5)

        if i != 0:
            ax[0, i].set_yticklabels([])
            ax[1, i].set_yticklabels([])

    ax[0, -1].legend(
        framealpha=0.5,
        loc="best",
        ncols=1,
        fontsize=legend_fontsize,
    )
    for i in range(0, n_panels):
        ax[0, i].set_ylim(1e-6, ylim_top)

    ax[0, 0].set_ylabel("n [Mpc$^{-3}$]", fontsize=fontsize)
    ax[1, 0].set_ylabel("n$_{" + data_label + "}$ / n$_{diffsky}$", fontsize=fontsize)
    fig.savefig(
        savedir
        + "_fit_z"
        + str(z_min)
        + "-"
        + str(z_max)
        + "_"
        + savedir.split("/")[-2]
        + ".pdf"
    )

    plt.show()


def plot_n_mags(
    dataset,
    data_label,
    param_collection1,
    label1,
    dimension_labels,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    suptitle,
    savedir,
    param_collection2=None,
    label2=None,
    lg_n_thresh=None,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=10000,
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    dataset_mags = dataset.mags
    data_sky_area_degsq = dataset.data_sky_area_degsq

    z_min, z_max = np.round(z_min, 2), np.round(z_max, 2)
    z_mask = (dataset_mags[:, -1] > z_min) & (dataset_mags[:, -1] < z_max)
    dataset_mags_z = dataset_mags[z_mask]
    data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=z_min, zhigh=z_max).value

    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        dataset.tcurves,
        z_phot_table,
    )
    obs_mags1, weights1 = mag_kern(
        ran_key,
        param_collection1,
        lc_data,
        dataset.mag_columns,
        dataset.mag_thresh_column,
        dataset.mag_thresh,
        dataset.frac_cat,
    )

    n_panels = obs_mags1.shape[1]

    if data_label == "SDSS":
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if data_label == "FENIKS":
        fig_width = 2.25 * n_panels
        fig_height = n_panels / 1.5

        fontsize = 2.25 * n_panels
        labelsize = 1.75 * n_panels
        legend_fontsize = 1.25 * n_panels

    fig, ax = plt.subplots(
        2,
        n_panels,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.subplots_adjust(
        left=0.05, hspace=0, top=0.875, right=0.99, bottom=0.15, wspace=0.0
    )
    fig.suptitle(
        suptitle + "   |   " + str(z_min) + " < z < " + str(z_max), fontsize=24
    )
    for i in range(0, n_panels):
        bins = np.linspace(
            dataset_mags_z[:, i].min(),
            dataset_mags_z[:, i].max(),
            20,
        )

        bin_centers = (bins[1:] + bins[:-1]) / 2
        ax[0, i].set_xlim(bins[0], bins[-1])
        ax[1, i].set_xlim(bins[0], bins[-1])

        n_data, bin_edges, _ = ax[0, i].hist(
            dataset_mags_z[:, i],
            weights=np.ones_like(dataset_mags_z[:, i]) * (1 / data_vol_mpc3),
            bins=bins,
            color="k",
            label=data_label,
            alpha=0.5,
        )

        n_diffsky, _, _ = ax[0, i].hist(
            obs_mags1[:, i],
            weights=weights1 * (1 / lc_data.lc_tot_vol_mpc3),
            bins=bins,
            color="deepskyblue",
            label=label1,
            alpha=0.5,
        )

        if i == n_panels - 1:
            ylim_top = 2 * n_diffsky.max()

        ax[0, i].set_yscale("log")
        ax[0, i].tick_params(axis="both", direction="in", labelsize=labelsize)

        offset = n_data / n_diffsky
        ax[1, i].plot(bin_centers, offset, lw=2.0, color="k")
        ax[1, i].set_ylim(0.09, 10.1)
        ax[1, i].set_yscale("log")
        ax[1, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
        ax[1, i].tick_params(axis="both", direction="in", labelsize=labelsize)
        ax_offset_yticks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
        ax[1, i].set_yticks(ax_offset_yticks)
        ax[1, i].set_yticklabels(["", "0.2", "0.5", "1", "2", "5", ""])
        ax[1, i].axhspan(
            ax_offset_yticks[2], ax_offset_yticks[4], color="orange", alpha=0.25
        )
        ax[1, i].axhspan(
            ax_offset_yticks[1], ax_offset_yticks[2], color="orange", alpha=0.5
        )
        ax[1, i].axhspan(
            ax_offset_yticks[4], ax_offset_yticks[5], color="orange", alpha=0.5
        )
        ax[1, i].axhspan(0, ax_offset_yticks[1], color="orange", alpha=0.8)
        ax[1, i].axhspan(ax_offset_yticks[5], 10, color="orange", alpha=0.8)
        ax[1, i].axhline(1, color="green", alpha=0.5, lw=5)

        if i != 0:
            ax[0, i].set_yticklabels([])
            ax[1, i].set_yticklabels([])

    ax[0, -1].legend(
        framealpha=0.5,
        loc="best",
        ncols=1,
        fontsize=legend_fontsize,
    )
    for i in range(0, n_panels):
        ax[0, i].set_ylim(1e-6, ylim_top)

    ax[0, 0].set_ylabel("n [Mpc$^{-3}$]", fontsize=fontsize)
    ax[1, 0].set_ylabel("n$_{" + data_label + "}$ / n$_{diffsky}$", fontsize=fontsize)
    fig.savefig(
        savedir
        + "_mags_z"
        + str(z_min)
        + "-"
        + str(z_max)
        + "_"
        + savedir.split("/")[-2]
        + ".pdf"
    )

    plt.show()
