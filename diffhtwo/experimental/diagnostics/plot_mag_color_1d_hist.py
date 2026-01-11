import jax.numpy as jnp
import numpy as np
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

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_n_ugriz(
    diffstarpop_params1,
    diffstarpop_params2,
    spspop_params1,
    spspop_params2,
    ssp_err_pop_params1,
    ssp_err_pop_params2,
    dataset,
    data_sky_area_degsq,
    tcurves,
    mag_column,
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
    lh_centroids,
    lg_n_data_err_lh,
    lgmp_min=10.0,
    sky_area_degsq=0.1,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """mc lightcone"""
    ran_key, lc_key = jran.split(ran_key, 2)
    lc_args = (lc_key, lgmp_min, zmin, zmax, sky_area_degsq, cosmo_params)
    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*lc_args)
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

    fig, ax = plt.subplots(1, 5, figsize=(12, 3))
    fig.subplots_adjust(left=0.1, hspace=0, top=0.9, right=0.99, bottom=0.2, wspace=0.0)
    fig.suptitle(title)

    color_bin_edges = np.arange(-0.5 - dmag / 2, 2.0, dmag)
    mag_bin_edges = np.arange(18.0 - dmag / 2, 26.0, dmag)

    weights1 = np.concatenate(
        [
            lc_phot1.weights_q * (1 / lc_vol_mpc3),
            lc_phot1.weights_smooth_ms * (1 / lc_vol_mpc3),
            lc_phot1.weights_bursty_ms * (1 / lc_vol_mpc3),
        ]
    )

    weights2 = np.concatenate(
        [
            lc_phot2.weights_q * (1 / lc_vol_mpc3),
            lc_phot2.weights_smooth_ms * (1 / lc_vol_mpc3),
            lc_phot2.weights_bursty_ms * (1 / lc_vol_mpc3),
        ]
    )
    data_weights = np.ones_like(dataset[:, 0]) / data_vol_mpc3
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

        ax[i].hist(
            obs_colors_mag1,
            weights=weights1,
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
            weights=weights2,
            bins=bins,
            histtype="step",
            color="green",
            alpha=0.7,
            lw=2,
            label=label2,
        )

        # data
        ax[i].hist(
            dataset[:, i],
            weights=data_weights,
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
        ax[i].set_ylim(1e-6, 1e-1)
        if i != 0:
            ax[i].set_yticklabels([])
    plt.savefig(saveAs)
    plt.show()

    # Output loss based on lh_centroids, not 1D histograms as above
    lg_n_thresh = -8
    lc_nhalos = np.ones(lc_halopop["logmp0"].shape)
    ran_key, n_key = jran.split(ran_key, 2)

    # 1
    u_diffstarpop_params1 = get_unbounded_diffstarpop_params(diffstarpop_params1)
    u_diffstarpop_theta1, u_diffstarpop_unravel = ravel_pytree(u_diffstarpop_params1)

    u_spspop_params1 = get_unbounded_spspop_params_tw_dust(spspop_params1)
    u_spspop_theta1, u_spspop_unravel = ravel_pytree(u_spspop_params1)

    u_ssp_err_pop_params1 = get_unbounded_ssperr_params(ssp_err_pop_params1)
    u_ssp_err_pop_theta1, u_ssp_err_pop_unravel = ravel_pytree(u_ssp_err_pop_params1)

    u_theta1 = (u_diffstarpop_theta1, u_spspop_theta1, u_ssp_err_pop_theta1)

    # 2
    u_diffstarpop_params2 = get_unbounded_diffstarpop_params(diffstarpop_params2)
    u_diffstarpop_theta2, u_diffstarpop_unravel = ravel_pytree(u_diffstarpop_params2)

    u_spspop_params2 = get_unbounded_spspop_params_tw_dust(spspop_params2)
    u_spspop_theta2, u_spspop_unravel = ravel_pytree(u_spspop_params2)

    u_ssp_err_pop_params2 = get_unbounded_ssperr_params(ssp_err_pop_params2)
    u_ssp_err_pop_theta2, u_ssp_err_pop_unravel = ravel_pytree(u_ssp_err_pop_params2)

    u_theta1 = (u_diffstarpop_theta1, u_spspop_theta1, u_ssp_err_pop_theta1)
    u_theta2 = (u_diffstarpop_theta2, u_spspop_theta2, u_ssp_err_pop_theta2)

    args = (
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
        DEFAULT_COSMOLOGY,
        FB,
    )

    loss1 = n_mag_opt._loss_kern(u_theta1, lg_n_data_err_lh, *args)
    loss2 = n_mag_opt._loss_kern(u_theta2, lg_n_data_err_lh, *args)

    print(loss1, loss2)


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
