import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ..data_loaders.load_minerva import PhotFilters, get_filt_indx
from ..kernels.phot_kern import get_colors_mags, mag_kern
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data

blue = "#1E90FF"  # DodgerBlue
orange = "#FF8C00"  # DarkOrange

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

BASE_PATH = Path(__file__).resolve().parent.parent
IGM_DRN = BASE_PATH / "data" / "igm"
IGM_BN = "igm_attenuation_minerva.h5"

try:
    import matplotlib.lines as mlines
    from matplotlib import pyplot as plt

    plt.rc("font", family="serif", serif=["Times New Roman"])

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

MINERVA_ANCHORS = [
    "#7C93D6",
    "#8DC8D3",
    "#A3DBB4",
    "#BCDC9F",
    "#DAD9AB",
    "#EACFA2",
    "#E9C19D",
    "#D99B9B",
]


def minerva_colors(n):
    cmap = LinearSegmentedColormap.from_list("minerva", MINERVA_ANCHORS)
    return [cmap(x) for x in np.linspace(0, 1, n)]


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

    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        dataset.col_idx,
        dataset.mag_idx,
        dataset.filter_info.mag_thresh,
        dataset.frac_cat,
    )

    n_panels = obs_color_mag.shape[1] - len(dataset.mag_idx)

    if "sdss" in data_label:
        fig_width = 3.0 * n_panels
        fig_height = n_panels

        fontsize = 4 * n_panels
        # labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if "feniks" in data_label:
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

    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        dataset.col_idx,
        dataset.mag_idx,
        dataset.filter_info.mag_thresh,
        dataset.frac_cat,
    )

    n_panels = obs_color_mag.shape[1]

    if "sdss" in data_label:
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if "feniks" in data_label:
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
        if i >= n_panels - len(dataset.mag_idx):
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

    if "sdss" in data_label:
        fig_width = 3.0 * n_panels
        fig_height = 1.5 * n_panels

        fontsize = 4 * n_panels
        labelsize = 3.25 * n_panels
        legend_fontsize = 3 * n_panels

    if "feniks" in data_label:
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
    sdss_dataset,
    feniks_dataset,
    run_label,
    param_collection,
    ran_key,
    zbins,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    num_halos=5000,
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    dmag=0.5,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    plt_show=True,
):
    band_colors = [
        "#001219",
        "#064D54",
        "#1B8686",
        "#5EB59D",
        "#95B481",
        "#BE8433",
        "#B55120",
        "#9B1D20",
    ]
    bands = [r"$u$", r"$g$", r"$r$", r"$i$", r"$z$", r"$J$", r"$H$", r"$K$"]

    fig_width = 7.1
    fig_height = 2.6

    fontsize = 10
    labelsize = 10
    legendsize = 8
    alpha = 0.95
    lw = 0.75
    s = 2.5
    ypad = 0.2

    n_z_bins = len(zbins)

    fig, ax = plt.subplots(
        1, n_z_bins, figsize=(fig_width, fig_height), constrained_layout=True
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))

    for zbin in range(len(zbins)):
        if zbin == 0:
            dataset = sdss_dataset
        else:
            dataset = feniks_dataset

        redshift = dataset.redshift
        mags = dataset.mags
        mag_sels = dataset.mag_sels
        parent_cut_idx = dataset.parent_cut_idx
        data_sky_area_degsq = dataset.data_sky_area_degsq
        mag_thresh = dataset.filter_info.mag_thresh
        n_bands = mags.shape[1]

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min, z_max = np.round(z_min, 2), np.round(z_max, 2)

        survey = "SDSS" if zbin == 0 else "FENIKS"
        ax[zbin].set_title(f"{z_min} < z < {z_max}\n{survey}", fontsize=fontsize)

        z_mask = (redshift > z_min) & (redshift < z_max)
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
        obs_mags, gal_weight, mag_weight, phot_kern_results = mag_kern(
            ran_key,
            param_collection,
            lc_data,
            mag_thresh,
        )

        shift_dex = 0.0
        d_shift_dex = 0.2
        xs, ys = [], []
        for i in range(n_bands):
            sel = mag_sels[:, i] * z_mask
            mag_band_z = mags[sel][:, i]
            bins = np.arange(
                mag_thresh[i][0],
                mag_thresh[i][1],
                dmag,
            )
            bin_centers = (bins[1:] + bins[:-1]) / 2

            oversample_factor = 1
            bins_diffsky = np.linspace(
                bins[0], bins[-1], (len(bins) - 1) * oversample_factor + 1
            )
            bin_diffsky_centers = (bins_diffsky[1:] + bins_diffsky[:-1]) / 2

            n_data, bin_edges = np.histogram(
                mag_band_z,
                weights=np.ones_like(mag_band_z) * (1 / data_vol_mpc3),
                bins=bins,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_data = np.log10(n_data) + shift_dex
                ax[zbin].scatter(
                    bin_centers,
                    y_data,
                    c=band_colors[i],
                    alpha=alpha,
                    s=s,
                )
            finite = np.isfinite(y_data)
            xs.append(bin_centers[finite])
            ys.append(y_data[finite])

            n_diffsky, _ = np.histogram(
                obs_mags[:, i],
                weights=gal_weight
                * mag_weight[:, parent_cut_idx]
                * mag_weight[:, i]
                * (1 / lc_data.lc_tot_vol_mpc3),
                bins=bins_diffsky,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ax[zbin].plot(
                    bin_diffsky_centers,
                    np.log10(n_diffsky) + shift_dex,
                    c=band_colors[i],
                    alpha=alpha,
                    label=bands[i],
                    lw=lw,
                )
            shift_dex += d_shift_dex

        ax[zbin].set_xticks(np.arange(10, 30, 2))
        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        x, y = np.concatenate(xs), np.concatenate(ys)
        ax[zbin].set_xlim(x.min() - dmag, x.max() + dmag)
        ax[zbin].set_ylim(y.min() - ypad, y.max() + ypad)

    ax[0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)

    ax[0].annotate(
        "",
        xy=(0.9, 0.3),
        xytext=(0.9, 0.06),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.75, color="k"),
    )
    ax[0].text(
        0.86,
        0.18,
        "shift by\n +0.2 dex",
        transform=ax[0].transAxes,
        ha="right",
        va="center",
        fontsize=legendsize - 2,
    )

    handles = [
        mlines.Line2D([], [], color=c, linewidth=6, solid_capstyle="butt", label=label)
        for c, label in zip(band_colors, bands)
    ]
    handles += [
        mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=3,
            label="data",
        ),
        mlines.Line2D([], [], linestyle="-", lw=1, color="gray", label="diffsky"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0, 1.0, 1, 0),
        mode="expand",
        frameon=False,
        fontsize=legendsize,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    fig.supxlabel("apparent magnitude [AB]")
    fig.savefig(
        savedir + "/" + run_label + "_app_mag_funcs.png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()


def _add_marker_legend(ax, label, **kwargs):
    handle = Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor="gray",
        markeredgecolor="none",
        markersize=3,
        label=label,
    )
    ax.legend(
        handles=[handle],
        loc="upper center",
        frameon=False,
        handletextpad=0.0,
        **kwargs,
    )


def plot_app_mag_funcs_minerva(
    minerva_phot,
    run_label,
    param_collection,
    ran_key,
    ssp_data,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    logmp_cutoff=10.0,
    num_halos=5000,
    apply_igm=True,
    igm_drn=IGM_DRN,
    igm_bn=IGM_BN,
    igm_filters_namedtuple=PhotFilters,
    igm_filter_prefix="minerva_",
    lc_sky_area_degsq=1000,
    n_z_phot_table=30,
    dmag=0.5,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    plt_show=True,
):
    fig_width = 7.1
    fig_height = 2.5

    fontsize = 10
    labelsize = 10
    legendsize = 8
    alpha = 0.95
    lw = 0.75
    s = 2.5
    d_shift_dex = 0.3
    ypad = 0.2

    zbins = minerva_phot.zbins
    redshift = minerva_phot.redshift
    mags = minerva_phot.mags
    sels = minerva_phot.sels
    frac_cats = minerva_phot.frac_cats
    parent_cut_idx = minerva_phot.parent_cut_idx
    mags_labels = minerva_phot.mags_labels
    data_sky_area_degsq = minerva_phot.data_sky_area_degsq

    n_z_bins = len(zbins)

    fig, ax = plt.subplots(
        1, n_z_bins, figsize=(fig_width, fig_height), constrained_layout=True
    )
    ax = np.atleast_1d(ax)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.85))

    for zbin in range(len(zbins)):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min, z_max = np.round(z_min, 2), np.round(z_max, 2)

        ax[zbin].set_title(str(z_min) + " < z < " + str(z_max), y=1)

        z_mask = (redshift > z_min) & (redshift < z_max)

        data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=z_min, zhigh=z_max).value

        z_phot_table = 10 ** jnp.linspace(
            np.log10(z_min), np.log10(z_max), n_z_phot_table
        )
        lc_args = (
            ran_key,
            num_halos,
            z_min,
            z_max,
            lgmp_min,
            lgmp_max,
            lc_sky_area_degsq,
            ssp_data,
            minerva_phot.filter_info.tcurves,
            z_phot_table,
        )

        lc_data = generate_lc_data(
            *lc_args,
            apply_igm=apply_igm,
            igm_drn=igm_drn,
            igm_bn=igm_bn,
            igm_filters_namedtuple=igm_filters_namedtuple,
            igm_filter_prefix=igm_filter_prefix,
            logmp_cutoff=lgmp_min,
        )

        obs_mags, gal_weight, mag_weight, phot_kern_results = mag_kern(
            ran_key,
            param_collection,
            lc_data,
            minerva_phot.filter_info.mag_thresh,
        )

        shift_dex = 0.0
        xs, ys = [], []

        fields = minerva_phot.spaces[zbin]._fields[4:]
        mag_filters = [f for f in fields if "_" not in f]

        n_filters = len(mag_filters)
        colors = minerva_colors(n_filters)

        band_colors_fitted = []
        mags_labels_fitted = []
        for i in range(n_filters):
            (mag_idx,) = get_filt_indx(mag_filters[i], PhotFilters)

            sel = sels[:, mag_idx] * z_mask
            mag_band_z = mags[sel][:, mag_idx]

            bins = np.arange(
                mag_band_z.min(),
                mag_band_z.max() + dmag,
                dmag,
            )
            bin_centers = (bins[1:] + bins[:-1]) / 2

            n_data, _ = np.histogram(
                mag_band_z,
                weights=np.ones_like(mag_band_z) * (1 / data_vol_mpc3),
                bins=bins,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_data = np.log10(n_data) + shift_dex
                ax[zbin].scatter(
                    bin_centers,
                    y_data,
                    color=colors[i],
                    alpha=alpha,
                    s=s,
                )
            finite = np.isfinite(y_data)
            xs.append(bin_centers[finite])
            ys.append(y_data[finite])

            n_diffsky, _ = np.histogram(
                obs_mags[:, mag_idx],
                weights=gal_weight
                * mag_weight[:, mag_idx]
                * mag_weight[:, parent_cut_idx]
                * (1 / lc_data.lc_tot_vol_mpc3)
                * frac_cats[mag_idx],
                bins=bins,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ax[zbin].plot(
                    bin_centers,
                    np.log10(n_diffsky) + shift_dex,
                    color=colors[i],
                    alpha=alpha,
                    lw=lw,
                )
            shift_dex += d_shift_dex
            band_colors_fitted.append(colors[i])
            mags_labels_fitted.append(mags_labels[mag_idx])

        ax[zbin].set_xticks(np.arange(10, 30, 2))
        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        x, y = np.concatenate(xs), np.concatenate(ys)
        ax[zbin].set_xlim(x.min() - dmag, x.max() + dmag)
        ax[zbin].set_ylim(y.min() - ypad, y.max() + ypad)

    ax[0].set_ylabel("log$_{10}$ (n [Mpc$^{-3}$])", fontsize=fontsize)

    ax[0].annotate(
        "",
        xy=(0.9, 0.3),
        xytext=(0.9, 0.06),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.75, color="k"),
    )
    ax[0].text(
        0.86,
        0.18,
        f"shift by\n +{d_shift_dex} dex",
        transform=ax[0].transAxes,
        ha="right",
        va="center",
        fontsize=legendsize - 2,
    )

    leg = fig.legend(
        handles=[
            Line2D([], [], linestyle="none", label=l)
            for l in mags_labels_fitted  # noqa: E741
        ],
        loc="center left",
        ncol=len(mags_labels_fitted),
        frameon=False,
        fontsize=legendsize,
        handlelength=0,
        handletextpad=0,
        columnspacing=0.8,
    )
    for t, c in zip(leg.get_texts(), band_colors_fitted):
        t.set_color("w")
        t.set_bbox(dict(facecolor=c, edgecolor="none", pad=2))

    leg2 = fig.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="none",
                markersize=3,
                label="MINERVA",
            ),
            Line2D([], [], linestyle="-", lw=1, color="gray", label="diffsky"),
        ],
        loc="center left",
        ncol=2,
        frameon=False,
        fontsize=legendsize,
    )

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    w1 = leg.get_window_extent().transformed(inv).width
    w2 = leg2.get_window_extent().transformed(inv).width
    gap = 0.03
    y = 0.93
    x0 = 0.5 - (w1 + gap + w2) / 2
    leg.set_bbox_to_anchor((x0, y), transform=fig.transFigure)
    leg2.set_bbox_to_anchor((x0 + w1 + gap, y), transform=fig.transFigure)

    fig.supxlabel("apparent magnitude [AB]")
    fig.savefig(
        savedir + "/" + run_label + "_minerva_app_mag_funcs.png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()
