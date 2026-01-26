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

try:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_n_mag_ugriz(
    diffstarpop_params1,
    diffstarpop_params2,
    spspop_params1,
    spspop_params2,
    ssp_err_pop_params1,
    ssp_err_pop_params2,
    dataset_mags,
    data_sky_area_degsq,
    tcurves,
    mag_column,
    mag_thresh,
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

    # set weights=0 for mag > mag_thresh for the band indicated by mag_column
    obs_mag_q1 = lc_phot1.obs_mags_q[:, mag_column]
    obs_mag_smooth_ms1 = lc_phot1.obs_mags_smooth_ms[:, mag_column]
    obs_mag_bursty_ms1 = lc_phot1.obs_mags_bursty_ms[:, mag_column]

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

    # set weights=0 for mag > mag_thresh for the band indicated by mag_column
    obs_mag_q2 = lc_phot2.obs_mags_q[:, mag_column]
    obs_mag_smooth_ms2 = lc_phot2.obs_mags_smooth_ms[:, mag_column]
    obs_mag_bursty_ms2 = lc_phot2.obs_mags_bursty_ms[:, mag_column]

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

    fig, ax = plt.subplots(1, 5, figsize=(12, 3))
    fig.subplots_adjust(left=0.1, hspace=0, top=0.9, right=0.99, bottom=0.2, wspace=0.0)
    fig.suptitle(title)

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
            weights=N_weights1 * (1 / lc_vol_mpc3),
            bins=mag_bin_edges,
            histtype="step",
            color="k",
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
            weights=N_weights2 * (1 / lc_vol_mpc3),
            bins=mag_bin_edges,
            histtype="step",
            color="green",
            alpha=0.7,
            lw=2,
            label=label2,
        )

        # data
        ax[i].hist(
            dataset_mags[:, i],
            weights=np.ones_like(dataset_mags[:, i]) / data_vol_mpc3,
            bins=mag_bin_edges,
            color="orange",
            alpha=0.7,
            label="data",
        )

        ax[i].set_yscale("log")

    ax[0].set_ylabel("number density [Mpc$^{-3}$]")
    ax[0].set_xlabel("MegaCam_uS [AB]")
    ax[1].set_xlabel("HSC_g [AB]")
    ax[2].set_xlabel("HSC_r [AB]")
    ax[3].set_xlabel("HSC_i [AB]")
    ax[4].set_xlabel("HSC_z [AB]")
    ax[4].legend(framealpha=0.5)

    for i in range(0, n_bands):
        ax[i].set_ylim(1e-6, 1e-2)
        if i != 0:
            ax[i].set_yticklabels([])
    plt.savefig(saveAs + ".pdf")
    plt.show()


def plot_n_ugriz(
    diffstarpop_params1,
    diffstarpop_params2,
    spspop_params1,
    spspop_params2,
    ssp_err_pop_params1,
    ssp_err_pop_params2,
    dataset_colors_mag,
    data_sky_area_degsq,
    tcurves,
    mag_column,
    mag_thresh,
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
    ) = get_obs_colors_mag(lc_phot1, mag_column)
    obs_colors_mag1 = np.concatenate(
        [obs_colors_mag_q1, obs_colors_mag_smooth_ms1, obs_colors_mag_bursty_ms1]
    )

    # set weights=0 for mag > mag_thresh for the band indicated by mag_column
    obs_mag_q1 = obs_colors_mag_q1[:, -1]
    obs_mag_smooth_ms1 = obs_colors_mag_smooth_ms1[:, -1]
    obs_mag_bursty_ms1 = obs_colors_mag_bursty_ms1[:, -1]

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
    ) = get_obs_colors_mag(lc_phot2, mag_column)
    obs_colors_mag2 = np.concatenate(
        [obs_colors_mag_q2, obs_colors_mag_smooth_ms2, obs_colors_mag_bursty_ms2]
    )
    # set weights=0 for mag > mag_thresh for the band indicated by mag_column
    obs_mag_q2 = obs_colors_mag_q2[:, -1]
    obs_mag_smooth_ms2 = obs_colors_mag_smooth_ms2[:, -1]
    obs_mag_bursty_ms2 = obs_colors_mag_bursty_ms2[:, -1]

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

    # Plot corner
    # ranges = [(0, 2.0), (0, 2.0), (-0.6, 1.0), (-0.6, 1.0), (18, mag_thresh)]
    labels = [
        r"$uS_{MegaCam} - g_{HSC} [AB]$",
        r"$g_{HSC} - r_{HSC} [AB]$",
        r"$r_{HSC} - i_{HSC} [AB]$",
        r"$i_{HSC} - z_{HSC} [AB]$",
        r"$i_{HSC} [AB]$",
    ]
    fig_corner = corner.corner(
        obs_colors_mag1,
        weights=N_weights1,
        labels=labels,
        color="k",
        # smooth=1.5,
        # bins=80,
        # smooth_1d=1.5,
        plot_datapoints=False,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "step", "alpha": 0.9, "density": True},
        fill_contours=False,
        # range=ranges,
    )

    corner.corner(
        obs_colors_mag2,
        weights=N_weights2,
        fig=fig_corner,
        color="green",
        # smooth=False,
        # bins=80,
        # smooth_1d=1.5,
        plot_datapoints=False,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "step", "alpha": 0.9, "lw": 2, "density": True},
        hist2d_kwargs={"weights": N_weights2},
        fill_contours=False,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        # range=ranges,
    )

    corner.corner(
        dataset_colors_mag,
        fig=fig_corner,
        color="orange",
        plot_datapoints=False,
        # smooth=1.5,
        # bins=80,
        # smooth_1d=1.5,
        levels=[0.68, 0.95],
        hist_kwargs={"histtype": "stepfilled", "alpha": 1.0, "density": True},
        fill_contours=False,
        # range=ranges,
    )
    # proxy artists
    handles = [
        Line2D([], [], color="k", lw=2, label=label1),
        Line2D([], [], color="green", lw=2, label=label2),
        Line2D([], [], color="orange", lw=2, label="data"),
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
    fig, ax = plt.subplots(1, 5, figsize=(12, 3))
    fig.subplots_adjust(left=0.1, hspace=0, top=0.9, right=0.99, bottom=0.2, wspace=0.0)
    fig.suptitle(title)

    color_bin_edges = np.arange(-0.5 - dmag / 2, 2.0, dmag)
    mag_bin_edges = np.arange(18.0 - dmag / 2, mag_thresh, dmag)

    for i in range(0, n_bands):
        if i == n_bands - 1:
            bins = mag_bin_edges
        else:
            bins = color_bin_edges

        obs_colors_mag1 = np.concatenate(
            [
                obs_colors_mag_q1[:, i],
                obs_colors_mag_smooth_ms1[:, i],
                obs_colors_mag_bursty_ms1[:, i],
            ]
        )

        # diffndhist
        # bins_lo = bins[:-1]
        # bins_hi = bins[1:]
        # bins_lo = bins_lo.reshape(bins_lo.size, 1)
        # bins_hi = bins_hi.reshape(bins_hi.size, 1)
        # dataset_colors_mag_i = dataset_colors_mag[:, i].reshape(dataset_colors_mag[:, i].size, 1)
        # dataset_colors_mag_sig_i = jnp.zeros_like(dataset_colors_mag_i)
        # dataset_colors_mag_weights_i = jnp.ones_like(dataset_colors_mag_i) / data_vol_mpc3

        # n_data1 = diffndhist.tw_ndhist_weighted(
        #     dataset_colors_mag_i,
        #     dataset_colors_mag_sig_i,
        #     dataset_colors_mag_weights_i,
        #     bins_lo,
        #     bins_hi,
        # )

        # # q1
        # obs_colors_mag_q1_i = obs_colors_mag_q1[:, i]
        # obs_colors_mag_q1_i = obs_colors_mag_q1_i.reshape(obs_colors_mag_q1_i.size, 1)
        # sig = jnp.zeros(obs_colors_mag_q1_i.shape) + (dmag / 8)
        # lc_phot1_weights_q_i = lc_phot1.weights_q.reshape(lc_phot1.weights_q.size, 1)

        # N_q1 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_q1_i,
        #     sig,
        #     lc_phot1_weights_q_i,
        #     bins_lo,
        #     bins_hi,
        # )

        # # smooth_ms1
        # obs_colors_mag_smooth_ms1_i = obs_colors_mag_smooth_ms1[:, i]
        # obs_colors_mag_smooth_ms1_i = obs_colors_mag_smooth_ms1_i.reshape(
        #     obs_colors_mag_smooth_ms1_i.size, 1
        # )
        # lc_phot1_weights_smooth_ms_i = lc_phot1.weights_smooth_ms.reshape(
        #     lc_phot1.weights_smooth_ms.size, 1
        # )

        # N_smooth_ms1 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_smooth_ms1_i,
        #     sig,
        #     lc_phot1_weights_smooth_ms_i,
        #     bins_lo,
        #     bins_hi,
        # )

        # # bursty_ms1
        # obs_colors_mag_bursty_ms1_i = obs_colors_mag_bursty_ms1[:, i]
        # obs_colors_mag_bursty_ms1_i = obs_colors_mag_bursty_ms1_i.reshape(
        #     obs_colors_mag_bursty_ms1_i.size, 1
        # )
        # lc_phot1_weights_bursty_ms_i = lc_phot1.weights_bursty_ms.reshape(
        #     lc_phot1.weights_bursty_ms.size, 1
        # )

        # N_bursty_ms1 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_bursty_ms1_i,
        #     sig,
        #     lc_phot1_weights_bursty_ms_i,
        #     bins_lo,
        #     bins_hi,
        # )
        # N1 = N_q1 + N_smooth_ms1 + N_bursty_ms1
        # n1 = N1 / lc_vol_mpc3

        # # q2
        # obs_colors_mag_q2_i = obs_colors_mag_q2[:, i]
        # obs_colors_mag_q2_i = obs_colors_mag_q2_i.reshape(obs_colors_mag_q2_i.size, 1)
        # sig = jnp.zeros(obs_colors_mag_q2_i.shape) + (dmag / 2)
        # lc_phot2_weights_q_i = lc_phot2.weights_q.reshape(lc_phot2.weights_q.size, 1)

        # N_q2 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_q2_i,
        #     sig,
        #     lc_phot2_weights_q_i,
        #     bins_lo,
        #     bins_hi,
        # )

        # # smooth_ms2
        # obs_colors_mag_smooth_ms2_i = obs_colors_mag_smooth_ms2[:, i]
        # obs_colors_mag_smooth_ms2_i = obs_colors_mag_smooth_ms2_i.reshape(
        #     obs_colors_mag_smooth_ms2_i.size, 1
        # )
        # lc_phot2_weights_smooth_ms_i = lc_phot2.weights_smooth_ms.reshape(
        #     lc_phot2.weights_smooth_ms.size, 1
        # )

        # N_smooth_ms2 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_smooth_ms2_i,
        #     sig,
        #     lc_phot2_weights_smooth_ms_i,
        #     bins_lo,
        #     bins_hi,
        # )

        # # bursty_ms2
        # obs_colors_mag_bursty_ms2_i = obs_colors_mag_bursty_ms2[:, i]
        # obs_colors_mag_bursty_ms2_i = obs_colors_mag_bursty_ms2_i.reshape(
        #     obs_colors_mag_bursty_ms2_i.size, 1
        # )
        # lc_phot2_weights_bursty_ms_i = lc_phot2.weights_bursty_ms.reshape(
        #     lc_phot2.weights_bursty_ms.size, 1
        # )

        # N_bursty_ms2 = diffndhist.tw_ndhist_weighted(
        #     obs_colors_mag_bursty_ms2_i,
        #     sig,
        #     lc_phot2_weights_bursty_ms_i,
        #     bins_lo,
        #     bins_hi,
        # )
        # N2 = N_q2 + N_smooth_ms2 + N_bursty_ms2
        # n2 = N2 / lc_vol_mpc3

        # data not weighted with volume
        # N_data = diffndhist.tw_ndhist(
        #     dataset_colors_mag_i,
        #     dataset_colors_mag_sig_i,
        #     bins_lo,
        #     bins_hi,
        # )
        # n_data2 = N_data / data_vol_mpc3

        # bin_centers = (bins[1:] + bins[:-1]) / 2
        # ax[i].scatter(bin_centers, n1, label=label1, c="k")
        # ax[i].scatter(bin_centers, n2, label=label2, c="green")
        # ax[i].scatter(
        #     bin_centers, n_data1, label="weighted" + label2, c="cyan", alpha=0.5
        # )
        # ax[i].scatter(bin_centers, n_data2, label=label2, c="magenta", alpha=0.5)

        ####

        ax[i].hist(
            obs_colors_mag1,
            weights=N_weights1 * (1 / lc_vol_mpc3),
            bins=bins,
            histtype="step",
            color="k",
            alpha=0.7,
            label=label1,
        )

        obs_colors_mag2 = np.concatenate(
            [
                obs_colors_mag_q2[:, i],
                obs_colors_mag_smooth_ms2[:, i],
                obs_colors_mag_bursty_ms2[:, i],
            ]
        )

        ax[i].hist(
            obs_colors_mag2,
            weights=N_weights2 * (1 / lc_vol_mpc3),
            bins=bins,
            histtype="step",
            color="green",
            alpha=0.7,
            lw=2,
            label=label2,
        )

        # data
        ax[i].hist(
            dataset_colors_mag[:, i],
            weights=np.ones_like(dataset_colors_mag[:, i]) / data_vol_mpc3,
            bins=bins,
            color="orange",
            alpha=0.7,
            label="data",
        )

        ax[i].set_yscale("log")

    ax[0].set_ylabel("number density [Mpc$^{-3}$]")
    ax[0].set_xlabel("MegaCam_uS - HSC_g [AB]")
    ax[1].set_xlabel("HSC_g- HSC_r [AB]")
    ax[2].set_xlabel("HSC_r - HSC_i [AB]")
    ax[3].set_xlabel("HSC_i - HSC_z [AB]")
    ax[4].set_xlabel("HSC_i [AB]")
    ax[4].legend()
    for i in range(0, n_bands):
        ax[i].set_ylim(1e-6, 2e-1)
        if i != 0:
            ax[i].set_yticklabels([])
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
            dmag,
            mag_column,
            mag_thresh,
            DEFAULT_COSMOLOGY,
            FB,
        )

        loss1 = n_mag_opt._loss_kern(u_theta1, lg_n_data_err_lh, *loss_args)
        loss2 = n_mag_opt._loss_kern(u_theta2, lg_n_data_err_lh, *loss_args)

        print(f"default loss = {loss1:.2f}")
        print(f"fit loss = {loss2:.2f}")


def get_obs_colors_mag(lc_phot, mag_column):
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
    obs_mag_q = lc_phot.obs_mags_q[:, mag_column]
    obs_colors_mag_q.append(obs_mag_q)
    obs_colors_mag_q = jnp.asarray(obs_colors_mag_q).T

    obs_mag_smooth_ms = lc_phot.obs_mags_smooth_ms[:, mag_column]
    obs_colors_mag_smooth_ms.append(obs_mag_smooth_ms)
    obs_colors_mag_smooth_ms = jnp.asarray(obs_colors_mag_smooth_ms).T

    obs_mag_bursty_ms = lc_phot.obs_mags_bursty_ms[:, mag_column]
    obs_colors_mag_bursty_ms.append(obs_mag_bursty_ms)
    obs_colors_mag_bursty_ms = jnp.asarray(obs_colors_mag_bursty_ms).T

    return obs_colors_mag_q, obs_colors_mag_smooth_ms, obs_colors_mag_bursty_ms
