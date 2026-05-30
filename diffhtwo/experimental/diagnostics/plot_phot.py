import warnings

import jax.numpy as jnp
import numpy as np
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..kernels.phot_kern import get_colors_mags, mag_kern
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data

blue = "#1E90FF"  # DodgerBlue
orange = "#FF8C00"  # DarkOrange
# blue = "#4169E1"  # RoyalBlue
# orange = "#D2691E"  # Chocolate
# blue = "#00BFFF"  # DeepSkyBlue
# orange = "#FFA500"  # Orange

mblue = "tab:blue"
morange = "tab:orange"
mred = "tab:red"

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
    import matplotlib.lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_color_pdfs(
    dataset,
    data_label,
    param_collection,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
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
        dataset.filter_info.tcurves,
        z_phot_table,
    )
    in_lh = jnp.array(list(dataset.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]
    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        dataset.filter_info.mag_thresh,
        in_lh_idx,
        dataset.frac_cat,
    )

    n_panels = obs_color_mag.shape[1] - len(in_lh_idx)

    if data_label == "sdss":
        fig_width = 3.0 * n_panels
        fig_height = n_panels

        fontsize = 4 * n_panels
        # labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if data_label == "feniks":
        fig_width = 2.25 * n_panels
        fig_height = n_panels / 2.5

        fontsize = 2.25 * n_panels
        # labelsize = 1.75 * n_panels
        legend_fontsize = 1.25 * n_panels

    fig, ax = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_height),
    )
    fig.subplots_adjust(
        left=0.05, hspace=0, top=0.875, right=0.99, bottom=0.15, wspace=0.0
    )
    fig.suptitle(str(z_min) + " < z < " + str(z_max), fontsize=20)
    for i in range(0, n_panels):
        std = np.std(dataset_colors_mag_z[:, i])
        med = np.median(dataset_colors_mag_z[:, i])
        bins = np.linspace(
            med - (6 * std),
            med + (6 * std),
            20,
        )

        # bin_centers = (bins[1:] + bins[:-1]) / 2
        ax[i].set_xlim(bins[0], bins[-1])
        ax[i].set_xlim(bins[0], bins[-1])
        ax[i].set_xlabel(dataset.dataset_dim_labels[i], fontsize=fontsize)

        n_data, bin_edges, _ = ax[i].hist(
            dataset_colors_mag_z[:, i],
            weights=np.ones_like(dataset_colors_mag_z[:, i]) * (1 / data_vol_mpc3),
            bins=bins,
            color="k",
            label=data_label,
            alpha=0.5,
            density=True,
        )

        n_diffsky, _, _ = ax[i].hist(
            obs_color_mag[:, i],
            weights=weights * (1 / lc_data.lc_tot_vol_mpc3),
            bins=bins,
            color="deepskyblue",
            label="diffsky",
            alpha=0.5,
            density=True,
        )

        ax[i].tick_params(
            which="major",
            length=0,
            # width=1.5,
            # direction="in",
            # top=True,
            # right=True,
            # labelsize=labelsize,
        )
        # ax[i].tick_params(which="minor", length=0, top=True, right=True)

        if i != 0:
            ax[i].set_yticklabels([])

    ax[-1].legend(
        framealpha=0.5,
        loc="best",
        ncols=1,
        fontsize=legend_fontsize,
    )

    ax[0].set_ylabel("PDF", fontsize=fontsize)
    fig.savefig(
        savedir
        + "/"
        + data_label
        + "_color_pdfs_z"
        + str(z_min)
        + "-"
        + str(z_max)
        + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def plot_n_colors_mag(
    dataset,
    data_label,
    param_collection,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
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
        dataset.filter_info.tcurves,
        z_phot_table,
    )
    in_lh = jnp.array(list(dataset.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]
    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        dataset.filter_info.mag_thresh,
        in_lh_idx,
        dataset.frac_cat,
    )

    n_panels = obs_color_mag.shape[1]

    if data_label == "sdss":
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if data_label == "feniks":
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
    fig.suptitle(str(z_min) + " < z < " + str(z_max), fontsize=24)
    for i in range(0, n_panels):
        if i >= n_panels - len(in_lh_idx):
            bins = np.linspace(
                dataset_colors_mag_z[:, i].min() - 0.2,
                dataset_colors_mag_z[:, i].max(),
                20,
            )
        else:
            std = np.std(dataset_colors_mag_z[:, i])
            med = np.median(dataset_colors_mag_z[:, i])
            bins = np.linspace(
                med - (6 * std),
                med + (6 * std),
                20,
            )

        bin_centers = (bins[1:] + bins[:-1]) / 2
        ax[0, i].set_xlim(bins[0], bins[-1])
        ax[0, i].set_xticks([])
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
            obs_color_mag[:, i],
            weights=weights * (1 / lc_data.lc_tot_vol_mpc3),
            bins=bins,
            color="deepskyblue",
            label="diffsky",
            alpha=0.5,
        )

        if i == n_panels - 1:
            ylim_top = 3 * n_diffsky.max()

        ax[0, i].set_yscale("log")

        ax[0, i].tick_params(
            which="major",
            length=6,
            width=1.5,
            direction="in",
            top=True,
            right=True,
            labelsize=labelsize,
        )
        ax[0, i].tick_params(
            which="minor", length=3, width=1.5, direction="in", top=True, right=True
        )
        ax[1, i].tick_params(
            which="major",
            length=6,
            width=1.5,
            direction="in",
            top=True,
            right=True,
            labelsize=labelsize,
        )
        ax[1, i].tick_params(
            which="minor", length=3, width=1.5, direction="in", top=True, right=True
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            offset = n_diffsky / n_data

        ax[1, i].plot(bin_centers, offset, lw=2.0, color="k")
        ax[1, i].set_ylim(0.09, 10.1)
        ax[1, i].set_yscale("log")
        ax[1, i].set_xlabel(dataset.dataset_dim_labels[i], fontsize=fontsize)

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
    ax[1, 0].set_ylabel("n$_{diffsky}$ / n$_{" + data_label + "}$", fontsize=fontsize)
    fig.savefig(
        savedir + "/" + data_label + "_fit_z" + str(z_min) + "-" + str(z_max) + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def plot_n_mags(
    dataset,
    data_label,
    param_collection,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
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
        dataset.filter_info.tcurves,
        z_phot_table,
    )
    obs_mags, weights, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        lc_data,
        dataset.filter_info.mag_thresh,
        dataset.frac_cat,
    )

    n_panels = obs_mags.shape[1]

    if data_label == "sdss":
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if data_label == "feniks":
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
    fig.suptitle(str(z_min) + " < z < " + str(z_max), fontsize=24)
    for i in range(0, n_panels):
        bins = np.linspace(
            dataset_mags_z[:, i].min(),
            dataset_mags_z[:, i].max(),
            20,
        )

        bin_centers = (bins[1:] + bins[:-1]) / 2
        ax[0, i].set_xlim(bins[0], bins[-1] + 0.2)
        ax[0, i].set_xticks([])
        ax[1, i].set_xlim(bins[0], bins[-1] + 0.2)

        n_data, bin_edges, _ = ax[0, i].hist(
            dataset_mags_z[:, i],
            weights=np.ones_like(dataset_mags_z[:, i]) * (1 / data_vol_mpc3),
            bins=bins,
            color="k",
            label=data_label,
            alpha=0.5,
        )

        n_diffsky, _, _ = ax[0, i].hist(
            obs_mags[:, i],
            weights=weights * (1 / lc_data.lc_tot_vol_mpc3),
            bins=bins,
            color="deepskyblue",
            label="diffsky",
            alpha=0.5,
        )

        if i == n_panels - 1:
            ylim_top = 2 * n_diffsky.max()

        ax[0, i].set_yscale("log")
        ax[0, i].tick_params(
            which="major",
            length=6,
            width=1.5,
            direction="in",
            top=True,
            right=True,
            labelsize=labelsize,
        )
        ax[0, i].tick_params(
            which="minor", length=3, width=1.5, direction="in", top=True, right=True
        )
        ax[1, i].tick_params(
            which="major",
            length=6,
            width=1.5,
            direction="in",
            top=True,
            right=True,
            labelsize=labelsize,
        )
        ax[1, i].tick_params(
            which="minor", length=3, width=1.5, direction="in", top=True, right=True
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            offset = n_diffsky / n_data

        ax[1, i].plot(bin_centers, offset, lw=2.0, color="k")
        ax[1, i].set_ylim(0.09, 10.1)
        ax[1, i].set_yscale("log")
        ax[1, i].set_xlabel(dataset.mags_labels[i], fontsize=fontsize)

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
    ax[1, 0].set_ylabel("n$_{diffsky}$ / n$_{" + data_label + "}$", fontsize=fontsize)
    fig.savefig(
        savedir + "/" + data_label + "_mags_z" + str(z_min) + "-" + str(z_max) + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def plot_app_mag_funcs(
    dataset,
    data_label,
    param_collection,
    ran_key,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    plt_show=True,
):
    dataset_mags = dataset.mags
    data_sky_area_degsq = dataset.data_sky_area_degsq

    feniks_zbins = np.array(
        [
            [0.2, 0.4],
            [0.5, 0.7],
            [0.9, 1.1],
            [1.2, 1.6],
            [1.8, 2.2],
            [2.2, 2.6],
            [2.6, 3.0],
        ]
    )
    labels_z = [" z = " + str(np.round(np.median(z), 2)) for z in feniks_zbins]

    # colors_z = [
    #     "#004c6d",
    #     "#3d3d8f",
    #     "#7b2f8e",
    #     "#b0206e",
    #     "#d94050",
    #     "#f06b38",
    #     "#f79d28",
    #     "#ffd166",
    # ]
    colors_z = [
        "#001219",
        "#003d52",
        "#0a7a80",
        "#40b0a0",
        "#80cca8",
        "#b8dfa0",
        "#dfd080",
        "#f5e882",
    ]
    fig_width = 7.1
    fig_height = 5

    fontsize = 10
    labelsize = 10
    legend_fontsize = 10
    alpha = 0.75
    s = 10

    fig, ax = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    fig.subplots_adjust(
        left=0.05, hspace=0.3, top=0.875, right=0.99, bottom=0.1, wspace=0.1
    )

    handles = [
        mlines.Line2D([], [], color=c, linewidth=6, solid_capstyle="butt", label=label)
        for c, label in zip(colors_z, labels_z)
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        handlelength=3,
        handleheight=0.5,
        columnspacing=0.8,
        handletextpad=0.1,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=7,
    )

    xlim = []
    for zbin in range(0, len(feniks_zbins)):
        z_min = feniks_zbins[zbin][0]
        z_max = feniks_zbins[zbin][1]

        z_min, z_max = np.round(z_min, 2), np.round(z_max, 2)
        z_mask = (dataset_mags[:, -1] > z_min) & (dataset_mags[:, -1] < z_max)
        dataset_mags_z = dataset_mags[z_mask]
        data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=z_min, zhigh=z_max).value

        z_phot_table = 10 ** jnp.linspace(
            np.log10(z_min), np.log10(z_max), n_z_phot_table
        )
        lc_data = generate_lc_data(
            ran_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            lc_sky_area_degsq,
            ssp_data,
            dataset.filter_info.tcurves,
            z_phot_table,
        )
        obs_mags, weights, phot_kern_results = mag_kern(
            ran_key,
            param_collection,
            lc_data,
            dataset.filter_info.mag_thresh,
            dataset.frac_cat,
        )

        n_bands = obs_mags.shape[1]

        row = 0
        col = 0
        dmag = 0.5
        for i in range(0, n_bands):
            bins = np.arange(
                dataset_mags_z[:, i].min(),
                dataset_mags_z[:, i].max() + dmag,
                dmag,
            )
            if zbin == 0:
                xlim.append([bins.min() - 0.5, bins.max() + 1])

            bin_centers = (bins[1:] + bins[:-1]) / 2
            ax[row, col].set_xlim(bins[0], bins[-1] + 0.2)
            # ax[0, i].set_xticks([])

            n_data, bin_edges = np.histogram(
                dataset_mags_z[:, i],
                weights=np.ones_like(dataset_mags_z[:, i]) * (1 / data_vol_mpc3),
                bins=bins,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ax[row, col].scatter(
                    bin_centers, np.log10(n_data), c=colors_z[zbin], alpha=alpha, s=s
                )

            (
                n_diffsky,
                _,
            ) = np.histogram(
                obs_mags[:, i],
                weights=weights * (1 / lc_data.lc_tot_vol_mpc3),
                bins=bins,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ax[row, col].plot(
                    bin_centers, np.log10(n_diffsky), c=colors_z[zbin], alpha=alpha
                )

            ax[row, col].set_xticks(np.arange(15, 30, 2))
            ax[row, col].tick_params(
                which="major",
                length=3,
                width=1.5,
                direction="in",
                top=True,
                right=True,
                labelsize=labelsize,
            )
            ax[row, col].tick_params(
                which="minor",
                length=1.5,
                width=1.5,
                direction="in",
                top=True,
                right=True,
            )

            ax[row, col].set_ylim(-6.5, -2.5)
            ax[row, col].set_xlim(xlim[i])
            ax[row, col].set_xlabel(dataset.mags_labels[i])

            if col != 0:
                ax[row, col].set_yticklabels([])

            if col == 3:
                row += 1
                col = 0
            else:
                col += 1

    # ax[0, -1].legend(
    #     framealpha=0.5,
    #     loc="best",
    #     ncols=1,
    #     fontsize=legend_fontsize,
    # )

    ax[0, 0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)
    ax[1, 0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)
    fig.savefig(
        savedir + "/" + data_label + "_app_mag_funcs.png",
        bbox_inches="tight",
        dpi=200,
    )
    if plt_show:
        plt.show()
    plt.close()
