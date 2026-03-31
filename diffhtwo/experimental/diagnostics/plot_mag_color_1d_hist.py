import corner
import jax.numpy as jnp
import numpy as np

# from diffsky import diffndhist
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffstar.defaults import FB, T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import random as jran

from ..utils import zbin_area, zbin_volume

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
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_n_mag(
    diffstarpop_params1,
    spspop_params1,
    ssp_err_pop_params1,
    label1,
    tcurves,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    dimension_labels,
    ran_key,
    zmins,
    zmaxs,
    ssp_data,
    mzr_params,
    scatter_params,
    suptitle,
    zbin_titles,
    savedir,
    dataset_mags=None,
    n_bands=None,
    data_sky_area_degsq=None,
    diffstarpop_params2=None,
    spspop_params2=None,
    ssp_err_pop_params2=None,
    label2=None,
    dmag=0.1,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_vol_mpc3=7e4,
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
        left=0.1, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig.suptitle(suptitle, fontsize=32)

    fig_offset, ax_offset = plt.subplots(
        n_zbins,
        n_bands,
        figsize=(fig_width, fig_height),
    )

    fig_offset.subplots_adjust(
        left=0.1, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig_offset.suptitle(suptitle, fontsize=32)

    dataset_mags_z1 = np.array(dataset_mags[0])

    for z in range(0, n_zbins):
        zmin = zmins[z]
        zmax = zmaxs[z]
        dataset_mags_z = np.array(dataset_mags[z])

        t = int(n_bands / 2)
        ax[z, t].set_title(zbin_titles[z], y=0.85, fontsize=labelsize)

        """mc lightcone"""
        ran_key, lc_key = jran.split(ran_key, 2)
        sky_area_degsq = zbin_area(lc_vol_mpc3, zlow=zmin, zhigh=zmax).value
        lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq)
        lc_halopop = mclh.mc_lightcone_host_halo_diffmah(
            *lc_args, cosmo_params=cosmo_params, lgmp_max=lgmp_max
        )

        if data_sky_area_degsq is not None:
            data_vol_mpc3 = zbin_volume(
                data_sky_area_degsq, zlow=zmin, zhigh=zmax
            ).value

        n_z_phot_table = 33

        if (zmin < 0.24) & (zmax > 0.24):
            nb_z = jnp.array([0.2445706, 0.40185568])
            nb816_zspan = np.linspace(nb_z[0] - 0.02, nb_z[0] + 0.02, 11)
            nb921_zspan = np.linspace(nb_z[1] - 0.02, nb_z[1] + 0.02, 11)
            z1_zspan = np.linspace(0.2, 0.5, 11)
            z_phot_table = np.concatenate((nb816_zspan, nb921_zspan, z1_zspan))
            z_phot_table.sort()
        else:
            z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)

        t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
        lgt0 = jnp.log10(t_0)
        t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

        precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
        )

        wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

        ran_key, phot_key1 = jran.split(ran_key, 2)
        phot_args1 = (
            phot_key1,
            lc_halopop["z_obs"],
            lc_halopop["t_obs"],
            lc_halopop["mah_params"],
            lc_halopop["logmp0"],
            t_table,
            ssp_data,
            precomputed_ssp_mag_table,
            z_phot_table,
            wave_eff_table,
            diffstarpop_params1,
            mzr_params,
            spspop_params1,
            scatter_params,
            ssp_err_pop_params1,
            cosmo_params,
            fb,
        )

        lc_phot1 = lc_phot_kern.multiband_lc_phot_kern(*phot_args1)
        if n_bands is None:
            num_halos, n_bands = lc_phot1.obs_mags_q.shape

        # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
        obs_mag_q1 = lc_phot1.obs_mags_q[:, mag_thresh_column]
        obs_mag_smooth_ms1 = lc_phot1.obs_mags_smooth_ms[:, mag_thresh_column]
        obs_mag_bursty_ms1 = lc_phot1.obs_mags_bursty_ms[:, mag_thresh_column]

        lc_phot_weights_q1 = jnp.where(
            obs_mag_q1 < mag_thresh,
            lc_phot1.weights_q,
            jnp.zeros_like(lc_phot1.weights_q),
        )
        lc_phot_weights_smooth_ms1 = jnp.where(
            obs_mag_smooth_ms1 < mag_thresh,
            lc_phot1.weights_smooth_ms,
            jnp.zeros_like(lc_phot1.weights_smooth_ms),
        )
        lc_phot_weights_bursty_ms1 = jnp.where(
            obs_mag_bursty_ms1 < mag_thresh,
            lc_phot1.weights_bursty_ms,
            jnp.zeros_like(lc_phot1.weights_bursty_ms),
        )
        N_weights1 = np.concatenate(
            [
                lc_phot_weights_q1 * frac_cat,
                lc_phot_weights_smooth_ms1 * frac_cat,
                lc_phot_weights_bursty_ms1 * frac_cat,
            ]
        )

        if diffstarpop_params2 is not None:
            ran_key, phot_key2 = jran.split(ran_key, 2)
            phot_args2 = (
                phot_key2,
                lc_halopop["z_obs"],
                lc_halopop["t_obs"],
                lc_halopop["mah_params"],
                lc_halopop["logmp0"],
                t_table,
                ssp_data,
                precomputed_ssp_mag_table,
                z_phot_table,
                wave_eff_table,
                diffstarpop_params2,
                mzr_params,
                spspop_params2,
                scatter_params,
                ssp_err_pop_params2,
                cosmo_params,
                fb,
            )

            lc_phot2 = lc_phot_kern.multiband_lc_phot_kern(*phot_args2)

            # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
            obs_mag_q2 = lc_phot2.obs_mags_q[:, mag_thresh_column]
            obs_mag_smooth_ms2 = lc_phot2.obs_mags_smooth_ms[:, mag_thresh_column]
            obs_mag_bursty_ms2 = lc_phot2.obs_mags_bursty_ms[:, mag_thresh_column]

            lc_phot_weights_q2 = jnp.where(
                obs_mag_q2 < mag_thresh,
                lc_phot2.weights_q,
                jnp.zeros_like(lc_phot2.weights_q),
            )
            lc_phot_weights_smooth_ms2 = jnp.where(
                obs_mag_smooth_ms2 < mag_thresh,
                lc_phot2.weights_smooth_ms,
                jnp.zeros_like(lc_phot2.weights_smooth_ms),
            )
            lc_phot_weights_bursty_ms2 = jnp.where(
                obs_mag_bursty_ms2 < mag_thresh,
                lc_phot2.weights_bursty_ms,
                jnp.zeros_like(lc_phot2.weights_bursty_ms),
            )
            N_weights2 = np.concatenate(
                [
                    lc_phot_weights_q2 * frac_cat,
                    lc_phot_weights_smooth_ms2 * frac_cat,
                    lc_phot_weights_bursty_ms2 * frac_cat,
                ]
            )

        for i in range(0, n_bands):
            ax_offset[z, i].axhline(+0.25, ls="--", lw=0.5, c="k")
            ax_offset[z, i].axhline(0, c="k")
            ax_offset[z, i].axhline(-0.25, ls="--", lw=0.5, c="k")

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

            # model 1
            lc_phot1_obs_mags = np.concatenate(
                [
                    lc_phot1.obs_mags_q[:, i],
                    lc_phot1.obs_mags_smooth_ms[:, i],
                    lc_phot1.obs_mags_bursty_ms[:, i],
                ]
            )
            lc_phot1_hist = ax[z, i].hist(
                lc_phot1_obs_mags,
                weights=N_weights1 * (1 / lc_vol_mpc3),
                bins=mag_bin_edges,
                histtype="step",
                color=color1,
                alpha=alpha1,
                label=label1,
                lw=lw + 1,
            )

            # model 2
            if diffstarpop_params2 is not None:
                lc_phot2_obs_mags = np.concatenate(
                    [
                        lc_phot2.obs_mags_q[:, i],
                        lc_phot2.obs_mags_smooth_ms[:, i],
                        lc_phot2.obs_mags_bursty_ms[:, i],
                    ]
                )

                lc_phot2_hist = ax[z, i].hist(
                    lc_phot2_obs_mags,
                    weights=N_weights2 * (1 / lc_vol_mpc3),
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
            offset_dex = np.log10(data_hist[0]) - np.log10(lc_phot1_hist[0])
            ax_offset[z, i].plot(mag_bin_centers, offset_dex, color=color1)
            ax_offset[z, i].set_ylim(-1, 1)

            ax[z, i].set_yscale("log")
            ax[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax_offset[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax[z, i].set_ylim(1e-6, 5e-3)
            ax[z, i].tick_params(axis="both", direction="in", labelsize=labelsize)
            if i != 0:
                ax[z, i].set_yticklabels([])
                ax_offset[z, i].set_yticklabels([])

    ax[0, -1].legend(
        framealpha=0.5,
        loc="upper left",
        bbox_to_anchor=(-2, 1.4),
        ncols=3,
        fontsize=legend_fontsize,
    )
    fig.supylabel("\u03d5 [Mpc$^{-3}$]", fontsize=fontsize)
    fig_offset.supylabel("log$_{10}$(n$_{FENIKS}$ / n$_{diffsky}$)", fontsize=fontsize)
    fig.savefig(savedir + "/mags_" + savedir.split("/")[-1] + ".pdf")
    fig_offset.savefig(savedir + "/mags_offsets_" + savedir.split("/")[-1] + ".pdf")
    plt.show()


def plot_n(
    diffstarpop_params1,
    spspop_params1,
    ssperrpop_params1,
    label1,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    dimension_labels,
    ran_key,
    zmins,
    zmaxs,
    ssp_data,
    mzr_params,
    scatter_params,
    suptitle,
    zbin_titles,
    savedir,
    dataset_colors_mag=None,
    data_sky_area_degsq=None,
    diffstarpop_params2=None,
    spspop_params2=None,
    ssperrpop_params2=None,
    label2=None,
    lh_centroids=None,
    lg_n_data_err_lh=None,
    lg_n_thresh=None,
    dmag=0.1,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_vol_mpc3=7e4,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    n_z_phot_table=15,
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
        left=0.05, hspace=0, top=0.95, right=0.99, bottom=0.05, wspace=0.0
    )
    fig.suptitle(suptitle, fontsize=32)
    dataset_colors_mag_z1 = np.array(dataset_colors_mag[0])
    for z in range(0, n_zbins):
        zmin = zmins[z]
        zmax = zmaxs[z]
        dataset_colors_mag_z = np.array(dataset_colors_mag[z])

        t = int(n_dims / 2)
        ax[z, t].set_title(zbin_titles[z], y=0.85, fontsize=labelsize)

        if data_sky_area_degsq is not None:
            data_vol_mpc3 = zbin_volume(
                data_sky_area_degsq, zlow=zmin, zhigh=zmax
            ).value

        if diffstarpop_params2 is not None:
            (
                obs_colors_mag1,
                N_weights1,
                obs_colors_mag2,
                N_weights2,
            ) = get_model_colors_mag(
                ran_key,
                diffstarpop_params1,
                spspop_params1,
                ssperrpop_params1,
                lc_vol_mpc3,
                zmin,
                zmax,
                tcurves,
                mag_columns,
                mag_thresh_column,
                mag_thresh,
                frac_cat,
                ssp_data,
                lgmp_min,
                lgmp_max,
                mzr_params,
                scatter_params,
                cosmo_params,
                fb,
                diffstarpop_params2=diffstarpop_params2,
                spspop_params2=spspop_params2,
                ssperrpop_params2=ssperrpop_params2,
            )

        else:
            obs_colors_mag1, N_weights1 = get_model_colors_mag(
                ran_key,
                diffstarpop_params1,
                spspop_params1,
                ssperrpop_params1,
                lc_vol_mpc3,
                zmin,
                zmax,
                tcurves,
                mag_columns,
                mag_thresh_column,
                mag_thresh,
                frac_cat,
                ssp_data,
                lgmp_min,
                lgmp_max,
                mzr_params,
                scatter_params,
                cosmo_params,
                fb,
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

            ax[z, i].hist(
                obs_colors_mag1[:, i],
                weights=N_weights1 * (1 / lc_vol_mpc3),
                bins=bins,
                histtype="step",
                color=color1,
                alpha=alpha1,
                lw=lw + 2,
                label=label1,
            )
            if diffstarpop_params2 is not None:
                ax[z, i].hist(
                    obs_colors_mag2[:, i],
                    weights=N_weights2 * (1 / lc_vol_mpc3),
                    bins=bins,
                    histtype="step",
                    color=color2,
                    alpha=alpha2,
                    lw=lw,
                    label=label2,
                )

            # data
            if dataset_colors_mag_z is not None:
                ax[z, i].hist(
                    dataset_colors_mag_z[:, i],
                    weights=np.ones_like(dataset_colors_mag_z[:, i])
                    * (1 / data_vol_mpc3),
                    bins=bins,
                    color=color_data,
                    alpha=alpha_data,
                    lw=lw,
                    label="FENIKS-UDS",
                )

            ax[z, i].set_yscale("log")
            ax[z, i].set_xlabel(dimension_labels[i], fontsize=fontsize)
            ax[z, i].set_ylim(1e-6, 3e-2)
            ax[z, i].tick_params(axis="both", direction="in", labelsize=labelsize)
            if i != 0:
                ax[z, i].set_yticklabels([])

        ax[z, 0].set_ylabel("\u03d5 [Mpc$^{-3}$]", fontsize=fontsize)
        ax[0, -1].legend(
            framealpha=0.5,
            loc="upper left",
            bbox_to_anchor=(-2, 1.4),
            ncols=3,
            fontsize=legend_fontsize,
        )

    plt.savefig(savedir + "/phot_fit_" + savedir.split("/")[-1] + ".pdf")
    plt.show()


def plot_n_corner(
    ran_key,
    diffstarpop_params1,
    spspop_params1,
    ssperrpop_params1,
    label1,
    zmin,
    zmax,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    dataset_colors_mag,
    dimension_labels,
    title,
    savedir,
    ssp_data,
    mzr_params,
    scatter_params,
    dmag=0.1,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_vol_mpc3=7e4,
    diffstarpop_params2=None,
    spspop_params2=None,
    ssperrpop_params2=None,
    label2=None,
):
    if diffstarpop_params2 is not None:
        (
            obs_colors_mag1,
            N_weights1,
            obs_colors_mag2,
            N_weights2,
        ) = get_model_colors_mag(
            ran_key,
            diffstarpop_params1,
            spspop_params1,
            ssperrpop_params1,
            lc_vol_mpc3,
            zmin,
            zmax,
            tcurves,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
            ssp_data,
            lgmp_min,
            lgmp_max,
            mzr_params,
            scatter_params,
            cosmo_params,
            fb,
            diffstarpop_params2=diffstarpop_params2,
            spspop_params2=spspop_params2,
            ssperrpop_params2=ssperrpop_params2,
        )

    else:
        obs_colors_mag1, N_weights1 = get_model_colors_mag(
            ran_key,
            diffstarpop_params1,
            spspop_params1,
            ssperrpop_params1,
            lc_vol_mpc3,
            zmin,
            zmax,
            tcurves,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
            ssp_data,
            lgmp_min,
            lgmp_max,
            mzr_params,
            scatter_params,
            cosmo_params,
            fb,
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
        # weights=dataset_colors_mag,
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
        obs_colors_mag1,
        weights=N_weights1,
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
    if diffstarpop_params2 is not None:
        corner.corner(
            obs_colors_mag2,
            weights=N_weights2,
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
        + str(zmin)
        + "-"
        + str(zmax)
        + "_corner_fit_"
        + savedir.split("/")[-1]
        + ".pdf"
    )
    plt.show()


def get_model_colors_mag(
    ran_key,
    diffstarpop_params1,
    spspop_params1,
    ssperrpop_params1,
    lc_vol_mpc3,
    zmin,
    zmax,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    ssp_data,
    lgmp_min,
    lgmp_max,
    mzr_params,
    scatter_params,
    cosmo_params,
    fb,
    n_z_phot_table=15,
    data_sky_area_degsq=None,
    diffstarpop_params2=None,
    spspop_params2=None,
    ssperrpop_params2=None,
):
    """mc lightcone"""
    ran_key, lc_key = jran.split(ran_key, 2)
    sky_area_degsq = zbin_area(lc_vol_mpc3, zlow=zmin, zhigh=zmax).value
    lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(
        *lc_args, cosmo_params=cosmo_params, lgmp_max=lgmp_max
    )

    z_phot_table = jnp.linspace(zmin, zmax, n_z_phot_table)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, cosmo_params
    )

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    ran_key, phot_key1 = jran.split(ran_key, 2)
    phot_args1 = (
        phot_key1,
        lc_halopop["z_obs"],
        lc_halopop["t_obs"],
        lc_halopop["mah_params"],
        lc_halopop["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params1,
        mzr_params,
        spspop_params1,
        scatter_params,
        ssperrpop_params1,
        cosmo_params,
        fb,
    )

    lc_phot1 = lc_phot_kern.multiband_lc_phot_kern(*phot_args1)
    num_halos, n_bands = lc_phot1.obs_mags_q.shape

    (
        obs_colors_mag_q1,
        obs_colors_mag_smooth_ms1,
        obs_colors_mag_bursty_ms1,
    ) = get_obs_colors_mag(lc_phot1, mag_columns)
    obs_colors_mag1 = np.concatenate(
        [obs_colors_mag_q1, obs_colors_mag_smooth_ms1, obs_colors_mag_bursty_ms1]
    )

    # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
    obs_mag_q1 = lc_phot1.obs_mags_q[:, mag_thresh_column]
    obs_mag_smooth_ms1 = lc_phot1.obs_mags_smooth_ms[:, mag_thresh_column]
    obs_mag_bursty_ms1 = lc_phot1.obs_mags_bursty_ms[:, mag_thresh_column]

    lc_phot_weights_q1 = jnp.where(
        obs_mag_q1 < mag_thresh,
        lc_phot1.weights_q,
        jnp.zeros_like(lc_phot1.weights_q),
    )
    lc_phot_weights_smooth_ms1 = jnp.where(
        obs_mag_smooth_ms1 < mag_thresh,
        lc_phot1.weights_smooth_ms,
        jnp.zeros_like(lc_phot1.weights_smooth_ms),
    )
    lc_phot_weights_bursty_ms1 = jnp.where(
        obs_mag_bursty_ms1 < mag_thresh,
        lc_phot1.weights_bursty_ms,
        jnp.zeros_like(lc_phot1.weights_bursty_ms),
    )
    N_weights1 = np.concatenate(
        [
            lc_phot_weights_q1 * frac_cat,
            lc_phot_weights_smooth_ms1 * frac_cat,
            lc_phot_weights_bursty_ms1 * frac_cat,
        ]
    )

    if diffstarpop_params2 is not None:
        ran_key, phot_key2 = jran.split(ran_key, 2)
        phot_args2 = (
            phot_key2,
            lc_halopop["z_obs"],
            lc_halopop["t_obs"],
            lc_halopop["mah_params"],
            lc_halopop["logmp0"],
            t_table,
            ssp_data,
            precomputed_ssp_mag_table,
            z_phot_table,
            wave_eff_table,
            diffstarpop_params2,
            mzr_params,
            spspop_params2,
            scatter_params,
            ssperrpop_params2,
            cosmo_params,
            fb,
        )

        lc_phot2 = lc_phot_kern.multiband_lc_phot_kern(*phot_args2)

        (
            obs_colors_mag_q2,
            obs_colors_mag_smooth_ms2,
            obs_colors_mag_bursty_ms2,
        ) = get_obs_colors_mag(lc_phot2, mag_columns)
        obs_colors_mag2 = np.concatenate(
            [
                obs_colors_mag_q2,
                obs_colors_mag_smooth_ms2,
                obs_colors_mag_bursty_ms2,
            ]
        )
        # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
        obs_mag_q2 = lc_phot2.obs_mags_q[:, mag_thresh_column]
        obs_mag_smooth_ms2 = lc_phot2.obs_mags_smooth_ms[:, mag_thresh_column]
        obs_mag_bursty_ms2 = lc_phot2.obs_mags_bursty_ms[:, mag_thresh_column]

        lc_phot_weights_q2 = jnp.where(
            obs_mag_q2 < mag_thresh,
            lc_phot2.weights_q,
            jnp.zeros_like(lc_phot2.weights_q),
        )
        lc_phot_weights_smooth_ms2 = jnp.where(
            obs_mag_smooth_ms2 < mag_thresh,
            lc_phot2.weights_smooth_ms,
            jnp.zeros_like(lc_phot2.weights_smooth_ms),
        )
        lc_phot_weights_bursty_ms2 = jnp.where(
            obs_mag_bursty_ms2 < mag_thresh,
            lc_phot2.weights_bursty_ms,
            jnp.zeros_like(lc_phot2.weights_bursty_ms),
        )
        N_weights2 = np.concatenate(
            [
                lc_phot_weights_q2 * frac_cat,
                lc_phot_weights_smooth_ms2 * frac_cat,
                lc_phot_weights_bursty_ms2 * frac_cat,
            ]
        )
        return obs_colors_mag1, N_weights1, obs_colors_mag2, N_weights2
    else:
        return obs_colors_mag1, N_weights1


def get_obs_colors_mag(lc_phot, mag_columns):
    num_halos, n_bands = lc_phot.obs_mags_q.shape

    obs_colors_mag_q = []
    obs_colors_mag_smooth_ms = []
    obs_colors_mag_bursty_ms = []

    for i in range(n_bands - 1):
        obs_color_q = lc_phot.obs_mags_q[:, i] - lc_phot.obs_mags_q[:, i + 1]
        obs_colors_mag_q.append(obs_color_q)

        obs_color_smooth_ms = (
            lc_phot.obs_mags_smooth_ms[:, i] - lc_phot.obs_mags_smooth_ms[:, i + 1]
        )
        obs_colors_mag_smooth_ms.append(obs_color_smooth_ms)

        obs_color_bursty_ms = (
            lc_phot.obs_mags_bursty_ms[:, i] - lc_phot.obs_mags_bursty_ms[:, i + 1]
        )
        obs_colors_mag_bursty_ms.append(obs_color_bursty_ms)

    """mag_column"""
    for mag_column in mag_columns:
        obs_mag_q = lc_phot.obs_mags_q[:, mag_column]
        obs_colors_mag_q.append(obs_mag_q)

        obs_mag_smooth_ms = lc_phot.obs_mags_smooth_ms[:, mag_column]
        obs_colors_mag_smooth_ms.append(obs_mag_smooth_ms)

        obs_mag_bursty_ms = lc_phot.obs_mags_bursty_ms[:, mag_column]
        obs_colors_mag_bursty_ms.append(obs_mag_bursty_ms)

    obs_colors_mag_q = jnp.asarray(obs_colors_mag_q).T
    obs_colors_mag_smooth_ms = jnp.asarray(obs_colors_mag_smooth_ms).T
    obs_colors_mag_bursty_ms = jnp.asarray(obs_colors_mag_bursty_ms).T

    return obs_colors_mag_q, obs_colors_mag_smooth_ms, obs_colors_mag_bursty_ms
