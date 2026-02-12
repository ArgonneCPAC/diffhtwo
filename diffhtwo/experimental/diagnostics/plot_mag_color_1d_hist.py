import corner
import jax.numpy as jnp
import numpy as np

# from diffsky import diffndhist
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import get_unbounded_spspop_params_tw_dust
from diffsky.ssp_err_model.defaults import get_unbounded_ssperr_params
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop import get_unbounded_diffstarpop_params
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity.umzr import DEFAULT_MZR_PARAMS
from jax import random as jran
from jax.flatten_util import ravel_pytree

from .. import n_mag_opt
from ..utils import zbin_volume

# from scipy.stats import gaussian_kde
# yellow7 = pplt.scale_luminance('yellow7', 1.05)


try:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_n_mag(
    diffstarpop_params1,
    diffstarpop_params2,
    spspop_params1,
    spspop_params2,
    ssp_err_pop_params1,
    ssp_err_pop_params2,
    dataset_mags,
    data_sky_area_degsq,
    tcurves,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    dimension_labels,
    dmag,
    ran_key,
    zmin,
    zmax,
    ssp_data,
    mzr_params,
    scatter_params,
    title,
    label1,
    label2,
    saveAs,
    lgmp_min=10.0,
    lgmp_max=15.0,
    sky_area_degsq=0.1,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """mc lightcone"""
    ran_key, lc_key = jran.split(ran_key, 2)
    lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(
        *lc_args, cosmo_params=cosmo_params, lgmp_max=lgmp_max
    )
    lc_vol_mpc3 = zbin_volume(sky_area_degsq, zlow=zmin, zhigh=zmax).value
    data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=zmin, zhigh=zmax).value

    n_z_phot_table = 15

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
    num_halos, n_bands = lc_phot1.obs_mags_q.shape

    # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
    obs_mag_q1 = lc_phot1.obs_mags_q[:, mag_thresh_column]
    obs_mag_smooth_ms1 = lc_phot1.obs_mags_smooth_ms[:, mag_thresh_column]
    obs_mag_bursty_ms1 = lc_phot1.obs_mags_bursty_ms[:, mag_thresh_column]

    lc_phot_weights_q1 = jnp.where(
        obs_mag_q1 < mag_thresh, lc_phot1.weights_q, jnp.zeros_like(lc_phot1.weights_q)
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
            lc_phot_weights_q1,
            lc_phot_weights_smooth_ms1,
            lc_phot_weights_bursty_ms1,
        ]
    )
    # correction added on 02/09/2026. The fraction of objects remaining after all bands
    # included have totflux !=-99.
    cat_weight = jnp.ones_like(N_weights1) * frac_cat
    ###################################################################

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
    num_halos, n_bands = lc_phot2.obs_mags_q.shape

    # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
    obs_mag_q2 = lc_phot2.obs_mags_q[:, mag_thresh_column]
    obs_mag_smooth_ms2 = lc_phot2.obs_mags_smooth_ms[:, mag_thresh_column]
    obs_mag_bursty_ms2 = lc_phot2.obs_mags_bursty_ms[:, mag_thresh_column]

    lc_phot_weights_q2 = jnp.where(
        obs_mag_q2 < mag_thresh, lc_phot2.weights_q, jnp.zeros_like(lc_phot2.weights_q)
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
            lc_phot_weights_q2,
            lc_phot_weights_smooth_ms2,
            lc_phot_weights_bursty_ms2,
        ]
    )

    fig, ax = plt.subplots(1, n_bands, figsize=(2.5 * n_bands, 4))
    fig.subplots_adjust(left=0.1, hspace=0, top=0.8, right=0.99, bottom=0.2, wspace=0.0)
    fig.suptitle(title, fontsize=18)

    mag_bin_edges = np.arange(18.0 - dmag / 2, mag_thresh, dmag)
    for i in range(0, n_bands):
        lc_phot1_obs_mags = np.concatenate(
            [
                lc_phot1.obs_mags_q[:, i],
                lc_phot1.obs_mags_smooth_ms[:, i],
                lc_phot1.obs_mags_bursty_ms[:, i],
            ]
        )
        ax[i].hist(
            lc_phot1_obs_mags,
            weights=N_weights1 * (1 / lc_vol_mpc3) * cat_weight,
            bins=mag_bin_edges,
            histtype="step",
            color="deepskyblue",
            alpha=0.7,
            label=label1,
        )

        lc_phot2_obs_mags = np.concatenate(
            [
                lc_phot2.obs_mags_q[:, i],
                lc_phot2.obs_mags_smooth_ms[:, i],
                lc_phot2.obs_mags_bursty_ms[:, i],
            ]
        )

        ax[i].hist(
            lc_phot2_obs_mags,
            weights=N_weights2 * (1 / lc_vol_mpc3) * cat_weight,
            bins=mag_bin_edges,
            histtype="step",
            color="magenta",
            alpha=0.7,
            lw=1,
            label=label2,
        )

        # data
        ax[i].hist(
            dataset_mags[:, i],
            weights=np.ones_like(dataset_mags[:, i]) / data_vol_mpc3,
            bins=mag_bin_edges,
            color="navajowhite",
            alpha=1,
            label="data",
        )

        ax[i].set_yscale("log")
        ax[i].set_xlabel(dimension_labels[i], fontsize=14)
        ax[i].set_ylim(1e-6, 1e-3)
        if i != 0:
            ax[i].set_yticklabels([])

    ax[0].set_ylabel("number density [Mpc$^{-3}$]", fontsize=14)
    plt.rcParams["legend.fontsize"] = 14
    ax[-1].legend(framealpha=0.5, loc="upper left", bbox_to_anchor=(-1, 1.2), ncols=3)
    plt.savefig(saveAs + ".pdf")
    plt.show()


def plot_n(
    diffstarpop_params1,
    diffstarpop_params2,
    spspop_params1,
    spspop_params2,
    ssp_err_pop_params1,
    ssp_err_pop_params2,
    dataset_colors_mag,
    data_sky_area_degsq,
    tcurves,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    dimension_labels,
    dmag,
    ran_key,
    zmin,
    zmax,
    ssp_data,
    mzr_params,
    scatter_params,
    title,
    label1,
    label2,
    saveAs,
    lh_centroids=None,
    lg_n_data_err_lh=None,
    lg_n_thresh=None,
    lgmp_min=10.0,
    lgmp_max=15.0,
    sky_area_degsq=0.1,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """mc lightcone"""
    ran_key, lc_key = jran.split(ran_key, 2)
    lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(
        *lc_args, cosmo_params=cosmo_params, lgmp_max=lgmp_max
    )
    lc_vol_mpc3 = zbin_volume(sky_area_degsq, zlow=zmin, zhigh=zmax).value
    data_vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=zmin, zhigh=zmax).value

    n_z_phot_table = 15

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
        obs_mag_q1 < mag_thresh, lc_phot1.weights_q, jnp.zeros_like(lc_phot1.weights_q)
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
            lc_phot_weights_q1,
            lc_phot_weights_smooth_ms1,
            lc_phot_weights_bursty_ms1,
        ]
    )
    # correction added on 02/09/2026. The fraction of objects remaining after all bands
    # included have totflux !=-99.
    cat_weight = jnp.ones_like(N_weights1) * frac_cat
    ###################################################################

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
    num_halos, n_bands = lc_phot2.obs_mags_q.shape

    (
        obs_colors_mag_q2,
        obs_colors_mag_smooth_ms2,
        obs_colors_mag_bursty_ms2,
    ) = get_obs_colors_mag(lc_phot2, mag_columns)
    obs_colors_mag2 = np.concatenate(
        [obs_colors_mag_q2, obs_colors_mag_smooth_ms2, obs_colors_mag_bursty_ms2]
    )
    # set weights=0 for mag > mag_thresh for the band indicated by mag_thresh_column
    obs_mag_q2 = lc_phot2.obs_mags_q[:, mag_thresh_column]
    obs_mag_smooth_ms2 = lc_phot2.obs_mags_smooth_ms[:, mag_thresh_column]
    obs_mag_bursty_ms2 = lc_phot2.obs_mags_bursty_ms[:, mag_thresh_column]

    lc_phot_weights_q2 = jnp.where(
        obs_mag_q2 < mag_thresh, lc_phot2.weights_q, jnp.zeros_like(lc_phot2.weights_q)
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
            lc_phot_weights_q2,
            lc_phot_weights_smooth_ms2,
            lc_phot_weights_bursty_ms2,
        ]
    )

    color_bin_edges = np.arange(-0.5 - dmag / 2, 2.0, dmag)
    mag_bin_edges = np.arange(18.0 - dmag / 2, mag_thresh, dmag)

    # Plot corner
    ranges = [(color_bin_edges[0], color_bin_edges[-1])] * (
        len(dimension_labels) - len(mag_columns)
    )
    for m in range(0, len(mag_columns)):
        ranges.append((mag_bin_edges[0], mag_bin_edges[-1]))

    fig_corner = corner.corner(
        obs_colors_mag1,
        weights=N_weights1 * cat_weight,
        labels=dimension_labels,
        color="deepskyblue",
        # smooth=1.5,
        # bins=80,
        # smooth_1d=2,
        plot_datapoints=False,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "step", "alpha": 0.9, "density": True},
        fill_contours=False,
        range=ranges,
    )

    fig_corner.suptitle(title)

    corner.corner(
        obs_colors_mag2,
        weights=N_weights2 * cat_weight,
        fig=fig_corner,
        color="magenta",
        # smooth=False,
        # bins=80,
        # smooth_1d=1.5,
        plot_datapoints=False,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "step", "alpha": 0.9, "lw": 1, "density": True},
        hist2d_kwargs={"weights": N_weights2},
        fill_contours=False,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        range=ranges,
    )

    corner.corner(
        dataset_colors_mag,
        fig=fig_corner,
        color="navajowhite",
        plot_datapoints=False,
        # smooth=1.5,
        # bins=80,
        # smooth_1d=1.5,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "stepfilled", "alpha": 1.0, "density": True},
        fill_contours=False,
        range=ranges,
    )
    # proxy artists
    handles = [
        Line2D([], [], color="deepskyblue", lw=1, label=label1),
        Line2D([], [], color="magenta", lw=1, label=label2),
        Line2D([], [], color="navajowhite", lw=1, label="data"),
    ]

    # attach legend to one axis (corner has many axes!)
    fig_corner.axes[0].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    plt.savefig(saveAs + "_corner.pdf")
    plt.show()

    # Plot 1D histograms
    fig, ax = plt.subplots(
        1,
        n_bands - 1 + len(mag_columns),
        figsize=(2.5 * n_bands - 1 + len(mag_columns), 4),
    )
    fig.subplots_adjust(left=0.1, hspace=0, top=0.8, right=0.99, bottom=0.2, wspace=0.0)
    fig.suptitle(title, fontsize=18)

    for i in range(0, n_bands - 1 + len(mag_columns)):
        if i < n_bands - 1:
            bins = color_bin_edges
        else:
            bins = mag_bin_edges

        ax[i].hist(
            obs_colors_mag1[:, i],
            weights=N_weights1 * (1 / lc_vol_mpc3) * cat_weight,
            bins=bins,
            histtype="step",
            color="deepskyblue",
            alpha=0.7,
            label=label1,
        )

        ax[i].hist(
            obs_colors_mag2[:, i],
            weights=N_weights2 * (1 / lc_vol_mpc3) * cat_weight,
            bins=bins,
            histtype="step",
            color="magenta",
            alpha=0.7,
            lw=1,
            label=label2,
        )

        # data
        ax[i].hist(
            dataset_colors_mag[:, i],
            weights=np.ones_like(dataset_colors_mag[:, i]) / data_vol_mpc3,
            bins=bins,
            color="navajowhite",
            alpha=1,
            label="data",
        )

        ax[i].set_yscale("log")
        ax[i].set_xlabel(dimension_labels[i], fontsize=14)
        ax[i].set_ylim(1e-6, 3e-2)
        if i != 0:
            ax[i].set_yticklabels([])

    ax[0].set_ylabel("number density [Mpc$^{-3}$]", fontsize=14)
    plt.rcParams["legend.fontsize"] = 14
    ax[-1].legend(framealpha=0.5, loc="upper left", bbox_to_anchor=(-1, 1.2), ncols=3)

    plt.savefig(saveAs + ".pdf")
    plt.show()

    # Output loss based on lh_centroids, not 1D histograms as above,
    # but the same lc_halopop
    if lh_centroids is not None:
        lc_nhalos = jnp.ones_like(lc_halopop["logmp0"])
        ran_key, n_key = jran.split(ran_key, 2)

        # 1
        u_diffstarpop_params1 = get_unbounded_diffstarpop_params(diffstarpop_params1)
        u_diffstarpop_theta1, u_diffstarpop_unravel = ravel_pytree(
            u_diffstarpop_params1
        )

        u_spspop_params1 = get_unbounded_spspop_params_tw_dust(spspop_params1)
        u_spspop_theta1, u_spspop_unravel = ravel_pytree(u_spspop_params1)

        u_ssp_err_pop_params1 = get_unbounded_ssperr_params(ssp_err_pop_params1)
        u_ssp_err_pop_theta1, u_ssp_err_pop_unravel = ravel_pytree(
            u_ssp_err_pop_params1
        )

        u_theta1 = (u_diffstarpop_theta1, u_spspop_theta1, u_ssp_err_pop_theta1)

        # 2
        u_diffstarpop_params2 = get_unbounded_diffstarpop_params(diffstarpop_params2)
        u_diffstarpop_theta2, u_diffstarpop_unravel = ravel_pytree(
            u_diffstarpop_params2
        )

        u_spspop_params2 = get_unbounded_spspop_params_tw_dust(spspop_params2)
        u_spspop_theta2, u_spspop_unravel = ravel_pytree(u_spspop_params2)

        u_ssp_err_pop_params2 = get_unbounded_ssperr_params(ssp_err_pop_params2)
        u_ssp_err_pop_theta2, u_ssp_err_pop_unravel = ravel_pytree(
            u_ssp_err_pop_params2
        )

        u_theta1 = (u_diffstarpop_theta1, u_spspop_theta1, u_ssp_err_pop_theta1)
        u_theta2 = (u_diffstarpop_theta2, u_spspop_theta2, u_ssp_err_pop_theta2)

        dmag_for_loss = 0.5
        loss_args = (
            lg_n_thresh,
            n_key,
            lc_halopop["z_obs"],
            lc_halopop["t_obs"],
            lc_halopop["mah_params"],
            lc_halopop["logmp0"],
            lc_nhalos,
            lc_vol_mpc3,
            t_table,
            ssp_data,
            precomputed_ssp_mag_table,
            z_phot_table,
            wave_eff_table,
            DEFAULT_MZR_PARAMS,
            DEFAULT_SCATTER_PARAMS,
            lh_centroids,
            dmag_for_loss,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            DEFAULT_COSMOLOGY,
            FB,
            frac_cat,
        )

        loss1 = n_mag_opt._loss_kern(u_theta1, lg_n_data_err_lh, *loss_args)
        loss2 = n_mag_opt._loss_kern(u_theta2, lg_n_data_err_lh, *loss_args)

        print(f"default loss = {loss1:.2f}")
        print(f"fit loss = {loss2:.2f}")


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
