from functools import partial

import jax.numpy as jnp
from diffsky.experimental.kernels import phot_kernels_merging as pkm
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from .. import diffndhist as diffndhist2
from .cat_weights import compute_cat_weights
from .gehrels_err import get_n_data_err
from .lc_phot_kern import mc_phot_kern_merging_wrapper


@jjit
def get_colors_mags(
    ran_key,
    param_collection,
    lc_data,
    mag_thresh,
    in_lh_idx,
    frac_cat,
):
    mags, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_thresh,
        frac_cat,
    )
    # collect colors and mags
    n_gals, n_bands = mags.shape
    obs_color_mag = mags[:, 0 : n_bands - 1] - mags[:, 1:n_bands]

    # beyond colors, additional lh dimensions holding apparent magnitudes
    mags_in_lh = mags[:, in_lh_idx]
    obs_color_mag = jnp.hstack((obs_color_mag, mags_in_lh))

    return obs_color_mag, gal_weight, phot_kern_results


@jjit
def mag_kern(
    ran_key,
    param_collection,
    lc_data,
    mag_thresh,
    frac_cat,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    phot_kern_results = mc_phot_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
    )
    obs_mags = phot_kern_results.obs_mags

    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    # update weights to incorporate mag thresh cuts and frac_cat
    gal_weight = compute_cat_weights(
        gal_weight, phot_kern_results, mag_thresh, frac_cat
    )

    return obs_mags, gal_weight, phot_kern_results


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def n_colors_mags_lh(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    redshift_as_last_dimension_in_lh=False,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    obs_color_mag, gal_weight, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        frac_cat,
    )

    # calculate number density in LH bins
    sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)

    if redshift_as_last_dimension_in_lh:
        z_obs = lc_data.z_obs.reshape(lc_data.z_obs.size, 1)
        obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

        N = diffndhist2.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            gal_weight,
            lh_centroids_lo,
            lh_centroids_hi,
        )
        lg_n, lg_n_avg_err = get_n_data_err(N, lc_data.lh_vol_mpc3)

    else:
        N = diffndhist2.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            gal_weight,
            lh_centroids_lo,
            lh_centroids_hi,
        )
        lg_n, lg_n_avg_err = get_n_data_err(N, lc_data.lc_tot_vol_mpc3)

    return lg_n, lg_n_avg_err


def get_mc_colors_mags(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
):
    mags, z_obs = monte_carlo_phot_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
    )
    # collect colors and mags
    n_gals, n_bands = mags.shape
    obs_color_mag = mags[:, 0 : n_bands - 1] - mags[:, 1:n_bands]
    for mag_column in mag_columns:
        mag = mags[:, mag_column][:, None]
        obs_color_mag = jnp.hstack((obs_color_mag, mag))

    obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

    return obs_color_mag


def monte_carlo_phot_kern(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    phot_kern_results, phot_randoms = pkm._mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *param_collection,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
        mc_merge,
    )
    obs_mags = phot_kern_results.obs_mags

    # apply mag thresh cut
    obs_mag_thresh_band = obs_mags[:, mag_thresh_column]
    mag_thresh_sel = obs_mag_thresh_band < mag_thresh
    obs_mags = obs_mags[mag_thresh_sel]

    z_obs = lc_data.z_obs.reshape(lc_data.z_obs.size, 1)
    z_obs = z_obs[mag_thresh_sel]

    return obs_mags, z_obs
