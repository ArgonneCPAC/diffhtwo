import corner
import jax.numpy as jnp
import numpy as np

# from diffsky import diffndhist
from diffsky.mass_functions import mc_hosts
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from .. import n_specphot
from ..utils import generate_lc_data, zbin_volume

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
fontsize = 24
labelsize = 20
legend_fontsize = 30


try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


mpl.rcParams["axes.linewidth"] = 2


def plot_n_mag(
    dataset_mags,
    data_sky_area_degsq,
    param_collection1,
    label1,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    line_wave_aa,
    dimension_labels,
    ran_key,
    zmins,
    zmaxs,
    ssp_data,
    suptitle,
    zbin_titles,
    savedir,
    param_collection2=None,
    label2=None,
    lg_n_thresh=None,
    dmag=0.1,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    num_halos=1000,
    sky_area_degsq=1.0,
    n_z_phot_table=15,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    # Plot 1D histograms
    n_bands = len(tcurves)
    n_zbins = len(zmins)

    fig_width = 3.0 * n_bands
    fig_height = 3.0 * n_zbins
    fig, ax = plt.subplots(
        n_zbins,
        n_bands,
        figsize=(fig_width, fig_height),
    )

    fig.subplots_adjust(
        left=0.065, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig.suptitle(suptitle, fontsize=32)

    fig_offset, ax_offset = plt.subplots(
        n_zbins,
        n_bands,
        figsize=(fig_width, fig_height),
    )

    fig_offset.subplots_adjust(
        left=0.065, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig_offset.suptitle(suptitle, fontsize=32)

    dataset_mags_z1 = np.array(dataset_mags[0])

    for z in range(0, n_zbins):
        z_min = zmins[z]
        z_max = zmaxs[z]
        dataset_mags_z = np.array(dataset_mags[z])

        t = int(n_bands / 2)
        ax[z, t].set_title(zbin_titles[z], y=0.85, fontsize=labelsize)

        if data_sky_area_degsq is not None:
            data_vol_mpc3 = zbin_volume(
                data_sky_area_degsq, zlow=z_min, zhigh=z_max
            ).value

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
            sky_area_degsq,
            ssp_data,
            tcurves,
            z_phot_table,
        )
        line_wave_table = jnp.array([line_wave_aa])
        obs_mag1, weights1 = n_specphot.n_phot_kern(
            ran_key,
            param_collection1,
            lc_data,
            line_wave_table,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
        )

        if param_collection2 is not None:
            obs_mag2, weights2 = n_specphot.n_phot_kern(
                ran_key,
                param_collection2,
                lc_data,
                line_wave_table,
                mag_columns,
                mag_thresh_column,
                mag_thresh,
                frac_cat,
            )

        for i in range(0, n_bands):
            sigma = np.std(dataset_mags_z1[:, i])
            lower_limit = np.mean(dataset_mags_z1[:, i]) - (4 * sigma)
            upper_limit = np.mean(dataset_mags_z1[:, i]) + (4 * sigma)
            if i == n_bands - 1:
                upper_limit = mag_thresh
            mag_bin_edges = np.arange(
                lower_limit,
                upper_limit,
                dmag,
            )
            ax[z, i].set_xlim(lower_limit, upper_limit)
            ax_offset[z, i].set_xlim(lower_limit, upper_limit)

            lc_phot1_hist = ax[z, i].hist(
                obs_mag1[:, i],
                weights=weights1 * (1 / lc_data.lc_vol_mpc3),
                bins=mag_bin_edges,
                histtype="step",
                color=color1,
                alpha=alpha1,
                label=label1,
                lw=lw + 1,
            )

            # model 2
            if param_collection2 is not None:
                ax[z, i].hist(
                    obs_mag2[:, i],
                    weights=weights2 * (1 / lc_data.lc_vol_mpc3),
                    bins=mag_bin_edges,
                    histtype="step",
                    color=color2,
                    alpha=alpha2,
                    lw=lw,
                    label=label2,
                )

            # data
            data_hist = ax[z, i].hist(
                dataset_mags_z[:, i],
                weights=np.ones_like(dataset_mags_z[:, i]) * (1 / data_vol_mpc3),
                bins=mag_bin_edges,
                color=color_data,
                lw=lw,
                alpha=alpha_data,
                label="FENIKS-UDS",
            )

            """ax_offset"""
            mag_bin_centers = (mag_bin_edges[1:] + mag_bin_edges[:-1]) / 2
            offset = data_hist[0] / lc_phot1_hist[0]
            ax_offset[z, i].plot(mag_bin_centers, offset, lw=2, color="k")
            ax_offset[z, i].set_ylim(0.09, 10.1)
            ax_offset[z, i].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax_offset[z, i].set_yscale("log")
            ax_offset[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax_offset[z, i].tick_params(
                axis="both", direction="in", labelsize=labelsize
            )

            ax[z, i].set_yscale("log")
            ax[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax[z, i].set_ylim(1e-6, 5e-3)
            ax[z, i].tick_params(axis="both", direction="in", labelsize=labelsize)

            ax_offset_yticks = np.array([0.2, 0.5, 1, 2, 5])
            ax_offset[z, i].set_yticks(ax_offset_yticks)
            ax_offset[z, i].axhspan(
                ax_offset_yticks[1], ax_offset_yticks[3], color="orange", alpha=0.5
            )
            ax_offset[z, i].axhspan(
                ax_offset_yticks[0], ax_offset_yticks[1], color="r", alpha=0.5
            )
            ax_offset[z, i].axhspan(
                ax_offset_yticks[3], ax_offset_yticks[4], color="r", alpha=0.5
            )
            ax_offset[z, i].axhspan(0, ax_offset_yticks[0], color="r", alpha=0.8)
            ax_offset[z, i].axhspan(ax_offset_yticks[4], 10, color="r", alpha=0.8)

            if i != 0:
                ax[z, i].set_yticklabels([])
                ax_offset[z, i].set_yticklabels([])
            if z != n_zbins - 1:
                ax[z, i].set_xticklabels([])
                ax_offset[z, i].set_xticklabels([])
            if i == 0:
                ax_offset[z, i].set_yticklabels(["5x", "2x", "1x", "2x", "5x"])

    ax[0, -1].legend(
        framealpha=0.5,
        loc="upper left",
        bbox_to_anchor=(-2, 1.4),
        ncols=3,
        fontsize=legend_fontsize,
    )
    fig.supylabel("\u03d5 [Mpc$^{-3}$]", fontsize=fontsize)
    fig.savefig(savedir + "/mags_" + savedir.split("/")[-1] + ".pdf")

    fig_offset.supylabel("n$_{FENIKS}$ / n$_{diffsky}$", fontsize=fontsize)
    fig_offset.savefig(savedir + "/mags_offsets_" + savedir.split("/")[-1] + ".pdf")

    plt.show()


def plot_n(
    dataset_colors_mag,
    data_sky_area_degsq,
    param_collection1,
    label1,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    line_wave_aa,
    dimension_labels,
    ran_key,
    zmins,
    zmaxs,
    ssp_data,
    suptitle,
    zbin_titles,
    savedir,
    param_collection2=None,
    label2=None,
    lg_n_thresh=None,
    dmag=0.1,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    num_halos=1000,
    sky_area_degsq=1.0,
    n_z_phot_table=15,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    # Plot 1D histograms
    n_bands = len(tcurves)
    n_dims = n_bands - 1 + len(mag_columns)
    n_zbins = len(zmins)

    fig_width = 3.00 * n_dims
    fig_height = 3.25 * n_zbins
    fig, ax = plt.subplots(
        n_zbins,
        n_dims,
        figsize=(fig_width, fig_height),
    )

    fig.subplots_adjust(
        left=0.065, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig.suptitle(suptitle, fontsize=32)

    fig_offset, ax_offset = plt.subplots(
        n_zbins,
        n_dims,
        figsize=(fig_width, fig_height),
    )

    fig_offset.subplots_adjust(
        left=0.065, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig_offset.suptitle(suptitle, fontsize=32)

    dataset_colors_mag_z1 = np.array(dataset_colors_mag[0])
    for z in range(0, n_zbins):
        z_min = zmins[z]
        z_max = zmaxs[z]
        dataset_colors_mag_z = np.array(dataset_colors_mag[z])

        t = int(n_dims / 2)
        ax[z, t].set_title(zbin_titles[z], y=0.85, fontsize=labelsize)

        if data_sky_area_degsq is not None:
            data_vol_mpc3 = zbin_volume(
                data_sky_area_degsq, zlow=z_min, zhigh=z_max
            ).value

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
                sky_area_degsq,
                ssp_data,
                tcurves,
                z_phot_table,
            )
            line_wave_table = jnp.array([line_wave_aa])
            obs_color_mag1, weights1 = n_specphot.n_colors_mags(
                ran_key,
                param_collection1,
                lc_data,
                line_wave_table,
                mag_columns,
                mag_thresh_column,
                mag_thresh,
                frac_cat,
            )

            if param_collection2 is not None:
                obs_color_mag2, weights2 = n_specphot.n_colors_mags(
                    ran_key,
                    param_collection2,
                    lc_data,
                    line_wave_table,
                    mag_columns,
                    mag_thresh_column,
                    mag_thresh,
                    frac_cat,
                )

        for i in range(0, n_dims):
            sigma = np.std(dataset_colors_mag_z1[:, i])
            lower_limit = np.mean(dataset_colors_mag_z1[:, i]) - (4 * sigma)
            upper_limit = np.mean(dataset_colors_mag_z1[:, i]) + (4 * sigma)
            if i == n_dims - 1:
                upper_limit = mag_thresh
            bins = np.arange(
                lower_limit,
                upper_limit,
                dmag,
            )
            ax[z, i].set_xlim(lower_limit, upper_limit)
            ax_offset[z, i].set_xlim(lower_limit, upper_limit)

            obs_colors_mag1_hist = ax[z, i].hist(
                obs_color_mag1[:, i],
                weights=weights1 * (1 / lc_data.lc_vol_mpc3),
                bins=bins,
                histtype="step",
                color=color1,
                alpha=alpha1,
                lw=lw + 2,
                label=label1,
            )
            if param_collection2 is not None:
                ax[z, i].hist(
                    obs_color_mag2[:, i],
                    weights=weights2 * (1 / lc_data.lc_vol_mpc3),
                    bins=bins,
                    histtype="step",
                    color=color2,
                    alpha=alpha2,
                    lw=lw,
                    label=label2,
                )

            # data
            if dataset_colors_mag_z is not None:
                dataset_colors_mag_hist = ax[z, i].hist(
                    dataset_colors_mag_z[:, i],
                    weights=np.ones_like(dataset_colors_mag_z[:, i])
                    * (1 / data_vol_mpc3),
                    bins=bins,
                    color=color_data,
                    alpha=alpha_data,
                    lw=lw,
                    label="FENIKS-UDS",
                )
            """ax_offset"""
            bin_centers = (bins[1:] + bins[:-1]) / 2
            offset = dataset_colors_mag_hist[0] / obs_colors_mag1_hist[0]
            ax_offset[z, i].plot(bin_centers, offset, lw=2, color="k")
            ax_offset[z, i].set_ylim(0.09, 10.1)
            ax_offset[z, i].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax_offset[z, i].set_yscale("log")
            ax_offset[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax_offset[z, i].tick_params(
                axis="both", direction="in", labelsize=labelsize
            )

            ax[z, i].set_yscale("log")
            ax[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax[z, i].set_ylim(1e-6, 3e-2)
            ax[z, i].tick_params(axis="both", direction="in", labelsize=labelsize)

            ax_offset_yticks = np.array([0.2, 0.5, 1, 2, 5])
            ax_offset[z, i].set_yticks(ax_offset_yticks)
            ax_offset[z, i].axhspan(
                ax_offset_yticks[1], ax_offset_yticks[3], color="orange", alpha=0.5
            )
            ax_offset[z, i].axhspan(
                ax_offset_yticks[0], ax_offset_yticks[1], color="r", alpha=0.5
            )
            ax_offset[z, i].axhspan(
                ax_offset_yticks[3], ax_offset_yticks[4], color="r", alpha=0.5
            )
            ax_offset[z, i].axhspan(0, ax_offset_yticks[0], color="r", alpha=0.8)
            ax_offset[z, i].axhspan(ax_offset_yticks[4], 10, color="r", alpha=0.8)

            if i != 0:
                ax[z, i].set_yticklabels([])
                ax_offset[z, i].set_yticklabels([])
            if z != n_zbins - 1:
                ax[z, i].set_xticklabels([])
                ax_offset[z, i].set_xticklabels([])
            if i == 0:
                ax_offset[z, i].set_yticklabels(["5x", "2x", "1x", "2x", "5x"])

    ax[0, -1].legend(
        framealpha=0.5,
        loc="upper left",
        bbox_to_anchor=(-2, 1.4),
        ncols=3,
        fontsize=legend_fontsize,
    )

    fig.supylabel("\u03d5 [Mpc$^{-3}$]", fontsize=fontsize)
    fig.savefig(savedir + "/phot_fit_" + savedir.split("/")[-1] + ".pdf")

    fig_offset.supylabel("n$_{FENIKS}$ / n$_{diffsky}$", fontsize=fontsize)
    fig_offset.savefig(savedir + "/phot_offsets_" + savedir.split("/")[-1] + ".pdf")

    plt.show()


def plot_n_corner(
    dataset_colors_mag,
    param_collection1,
    label1,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    line_wave_aa,
    dimension_labels,
    ran_key,
    z_min,
    z_max,
    ssp_data,
    title,
    savedir,
    param_collection2=None,
    label2=None,
    lg_n_thresh=None,
    dmag=0.1,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    num_halos=1000,
    sky_area_degsq=1.0,
    n_z_phot_table=15,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    line_wave_table = jnp.array([line_wave_aa])
    if param_collection2 is not None:
        obs_color_mag1, weights1 = n_specphot.n_colors_mags(
            ran_key,
            param_collection1,
            lc_data,
            line_wave_table,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
        )
        obs_color_mag2, weights2 = n_specphot.n_colors_mags(
            ran_key,
            param_collection2,
            lc_data,
            line_wave_table,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
        )

    else:
        obs_color_mag1, weights1 = n_specphot.n_colors_mags(
            ran_key,
            param_collection1,
            lc_data,
            line_wave_table,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
        )
    color_bin_edges = np.arange(-0.5 - dmag / 2, 2.2, dmag)
    mag_bin_edges = np.arange(18.0 - dmag / 2, mag_thresh, dmag)
    ranges = [(color_bin_edges[0], color_bin_edges[-1])] * (
        len(dimension_labels) - len(mag_columns)
    )
    for m in range(0, len(mag_columns)):
        ranges.append((mag_bin_edges[0], mag_bin_edges[-1]))

    # data
    fig_corner = corner.corner(
        dataset_colors_mag,
        color=color_data,
        labels=dimension_labels,
        label_kwargs={"fontsize": 20},
        plot_datapoints=False,
        smooth=1.0,
        levels=[0.68, 0.95],
        hist_kwargs={
            "histtype": "stepfilled",
            "alpha": 0.5,
            "lw": lw,
            "density": True,
        },
        fill_contours=False,
        plot_density=False,
        contour_kwargs={"linewidths": 3.5, "alpha": 0.75},
        range=ranges,
    )

    fig_corner.suptitle(title, fontsize=fontsize + 4)

    # model 1
    corner.corner(
        obs_color_mag1,
        weights=weights1,
        fig=fig_corner,
        color=color1,
        smooth=1.0,
        plot_datapoints=False,
        levels=[0.68, 0.95],
        hist_kwargs={
            "histtype": "step",
            "alpha": 0.5,
            "lw": lw + 2,
            "density": True,
        },
        fill_contours=False,
        plot_density=False,
        contour_kwargs={"linewidths": 3.5, "alpha": 0.75},
        range=ranges,
    )

    # model 2
    if param_collection2 is not None:
        corner.corner(
            obs_color_mag2,
            weights=weights2,
            fig=fig_corner,
            color=color2,
            smooth=1.0,
            plot_datapoints=False,
            levels=[0.68, 0.95],
            hist_kwargs={
                "histtype": "step",
                "alpha": 0.5,
                "lw": lw + 2,
                "density": True,
            },
            plot_density=False,
            fill_contours=False,
            contour_kwargs={"linewidths": 3.5, "alpha": 0.75},
            range=ranges,
        )

    if label2 is not None:
        handles = [
            Line2D([], [], color=color1, lw=lw + 1, label=label1),
            Line2D([], [], color=color2, lw=lw + 1, label=label2),
            Line2D([], [], color=color_data, lw=lw, label="FENIKS-UDS"),
        ]
    else:
        handles = [
            Line2D([], [], color=color1, lw=lw + 1, label=label1),
            Line2D([], [], color=color_data, lw=lw, label="FENIKS-UDS"),
        ]

    fig_corner.axes[0].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=fontsize,
    )

    for ax in fig_corner.get_axes():
        ax.tick_params(axis="both", direction="in", labelsize=labelsize / 2)
    fig_corner.savefig(
        savedir
        + "/z"
        + str(z_min)
        + "-"
        + str(z_max)
        + "_corner_fit_"
        + savedir.split("/")[-1]
        + ".pdf"
    )
    plt.show()
