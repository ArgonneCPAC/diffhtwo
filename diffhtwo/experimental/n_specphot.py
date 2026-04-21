from functools import partial

import jax.numpy as jnp
from diffsky.burstpop import freqburst_mono
from diffsky.experimental import mc_diffstarpop_wrappers as mcdw
from diffsky.experimental.kernels import mc_randoms
from diffsky.experimental.kernels import phot_kernels_merging as pkm
from diffsky.experimental.kernels import specphot_kernels_merging as spkm
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from . import diffndhist as diffndhist2
from . import emline_luminosity
from .n_mag import N_0, N_FLOOR, get_n_data_err


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
):
    obs_color_mag, weights = get_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        frac_cat,
    )

    if redshift_as_last_dimension_in_lh is True:
        z_obs = lc_data.z_obs.reshape(lc_data.z_obs.size, 1)
        obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

    # calculate number density in LH bins
    sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)

    N = diffndhist2.tw_ndhist_weighted(
        obs_color_mag,
        sig,
        weights,
        lh_centroids_lo,
        lh_centroids_hi,
    )
    lg_n, lg_n_avg_err = get_n_data_err(N, lc_data.lc_vol_mpc3)

    return lg_n, lg_n_avg_err


@jjit
def get_colors_mags(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
):
    mags, weights = phot_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        frac_cat,
    )
    # collect colors and mags
    n_gals, n_bands = mags.shape
    obs_color_mag = mags[:, 0 : n_bands - 1] - mags[:, 1:n_bands]
    for mag_column in mag_columns:
        mag = mags[:, mag_column][:, None]
        obs_color_mag = jnp.hstack((obs_color_mag, mag))

    return obs_color_mag, weights


@jjit
def phot_kern(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
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

    # get weights incorporating frac_cat
    weights = lc_data.nhalos * frac_cat

    # apply mag thresh cut
    obs_mag_thresh_band = obs_mags[:, mag_thresh_column]
    weights = jnp.where(
        obs_mag_thresh_band < mag_thresh, weights, jnp.zeros_like(weights)
    )

    return obs_mags, weights


@jjit
def n_spec_kern(
    ran_key,
    param_collection,
    lc_data,
    line_wave_table,
    lg_emline_Lbin_edges,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    _res = spkm._mc_specphot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
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

    (phot_kern_results, phot_randoms, spec_kern_results) = _res
    linelum_gal = spec_kern_results.linelum_gal

    sig = jnp.diff(lg_emline_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    _, emline_N = emline_luminosity.get_emline_luminosity_func(
        linelum_gal,
        lc_data.nhalos,
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    # take care of bins with low/zero number counts in a similar way to n_mag.get_n_data_err(), using same N_floor and N_0:
    emline_N = jnp.where(emline_N > N_FLOOR, emline_N, N_0)

    lg_emline_LF_model = jnp.log10(emline_N / lc_data.lc_vol_mpc3)

    return lg_emline_LF_model


def n_spec_q_ms_burst(
    ran_key,
    param_collection,
    lc_data,
    line_wave_table,
    lg_emline_Lbin_edges,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    n_t_table=100,
    mc_merge=0,
):
    # get randoms
    phot_randoms, sfh_params, merging_randoms = mc_randoms.get_mc_phot_merge_randoms(
        ran_key, param_collection.diffstarpop_params, lc_data.mah_params, cosmo_params
    )

    # get logsm_obs, logssfr_obs from diffstarpop
    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        lc_data.mah_params, sfh_params, lc_data.t_obs, cosmo_params, fb, n_t_table
    )

    # get p_burst for logsm_obs, logssfr_obs and freqburst_params
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        logsm_obs,
        logssfr_obs,
    )

    # get mc_is for q, ms, and burst
    mc_is_q = phot_randoms.mc_is_q
    mc_is_ms = ~mc_is_q

    mc_is_burst = phot_randoms.uran_pburst < p_burst
    mc_is_burst = (mc_is_ms) & (mc_is_burst)
    mc_is_ms = (mc_is_ms) & (~mc_is_burst)

    _res = spkm._mc_specphot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
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

    (phot_kern_results, _, spec_kern_results) = _res
    linelum_gal = spec_kern_results.linelum_gal

    sig = jnp.diff(lg_emline_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)

    # composite
    _, emline_N = emline_luminosity.get_emline_luminosity_func(
        linelum_gal,
        lc_data.nhalos,
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    # take care of bins with low/zero number counts in a similar way
    # to n_mag.get_n_data_err(), using same N_floor and N_0:
    emline_N = jnp.where(emline_N > N_FLOOR, emline_N, N_0)
    lg_emline_LF = jnp.log10(emline_N / lc_data.lc_vol_mpc3)

    # q
    linelums_q = linelum_gal[mc_is_q]
    _, emline_N_q = emline_luminosity.get_emline_luminosity_func(
        linelums_q,
        lc_data.nhalos[mc_is_q],
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    emline_N_q = jnp.where(emline_N_q > N_FLOOR, emline_N_q, N_0)
    lg_emline_LF_q = jnp.log10(emline_N_q / lc_data.lc_vol_mpc3)

    # ms
    linelums_ms = linelum_gal[mc_is_ms]
    _, emline_N_ms = emline_luminosity.get_emline_luminosity_func(
        linelums_ms,
        lc_data.nhalos[mc_is_ms],
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    emline_N_ms = jnp.where(emline_N_ms > N_FLOOR, emline_N_ms, N_0)
    lg_emline_LF_ms = jnp.log10(emline_N_ms / lc_data.lc_vol_mpc3)

    # burst
    linelums_burst = linelum_gal[mc_is_burst]
    _, emline_N_burst = emline_luminosity.get_emline_luminosity_func(
        linelums_burst,
        lc_data.nhalos[mc_is_burst],
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    emline_N_burst = jnp.where(emline_N_burst > N_FLOOR, emline_N_burst, N_0)
    lg_emline_LF_burst = jnp.log10(emline_N_burst / lc_data.lc_vol_mpc3)

    return lg_emline_LF, lg_emline_LF_q, lg_emline_LF_ms, lg_emline_LF_burst
