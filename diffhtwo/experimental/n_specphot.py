import jax.numpy as jnp
from diffsky.experimental.kernels import mc_phot_kernels as mcpk
from diffsky.merging import merging_model
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit
from jax.debug import print

from . import diffndhist as diffndhist2
from . import emline_luminosity
from .n_mag import N_0, N_FLOOR, get_n_data_err


@jjit
def n_phot(
    ran_key,
    param_collection,
    lc_data,
    line_wave_table,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    dmag_centroids,
    frac_cat,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        ran_key, param_collection[0], lc_data.mah_params, cosmo_params
    )

    _res = mcpk._mc_specphot_kern_merging(
        ran_key,
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *param_collection,
        merging_model.DEFAULT_MERGE_PARAMS,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
    )

    (
        phot_kern_results,
        linelums_in_situ,
        phot_randoms,
        flux_in_plus_ex_situ,
        merge_prob,
        mstar_obs,
        linelums_in_plus_ex_situ,
    ) = _res
    mags_in_plus_ex_situ = -2.5 * jnp.log10(flux_in_plus_ex_situ)

    # collect colors and mags
    n_gals, n_bands = mags_in_plus_ex_situ.shape
    obs_color_mag = (
        mags_in_plus_ex_situ[:, 0 : n_bands - 1] - mags_in_plus_ex_situ[:, 1:n_bands]
    )
    for mag_column in mag_columns:
        mag = mags_in_plus_ex_situ[:, mag_column][:, None]
        obs_color_mag = jnp.hstack((obs_color_mag, mag))

    # get weights
    weights = lc_data.nhalos * frac_cat

    # apply mag thresh cut
    obs_mag_thresh_band = mags_in_plus_ex_situ[:, mag_thresh_column]
    weights_threshd = jnp.where(
        obs_mag_thresh_band < mag_thresh, weights, jnp.zeros_like(weights)
    )

    # calculate number density in LH bins
    sig = jnp.zeros(lh_centroids.shape) + (dmag_centroids / 2)
    lh_centroids_lo = lh_centroids - (dmag_centroids / 2)
    lh_centroids_hi = lh_centroids + (dmag_centroids / 2)
    frac_cat = jnp.ones(n_gals)
    N = diffndhist2.tw_ndhist_weighted(
        obs_color_mag,
        sig,
        weights_threshd,
        lh_centroids_lo,
        lh_centroids_hi,
    )
    lg_n, lg_n_avg_err = get_n_data_err(N, lc_data.lc_vol_mpc3)

    return lg_n, lg_n_avg_err


@jjit
def n_spec(
    ran_key,
    param_collection,
    lc_data,
    line_wave_table,
    lg_emline_Lbin_edges,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        ran_key, param_collection[0], lc_data.mah_params, cosmo_params
    )

    _res = mcpk._mc_specphot_kern_merging(
        ran_key,
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *param_collection,
        merging_model.DEFAULT_MERGE_PARAMS,
        cosmo_params,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
    )

    (
        phot_kern_results,
        linelums_in_situ,
        phot_randoms,
        flux_in_plus_ex_situ,
        merge_prob,
        mstar_obs,
        linelums_in_plus_ex_situ,
    ) = _res

    sig = jnp.diff(lg_emline_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    _, emline_N = emline_luminosity.get_emline_luminosity_func(
        linelums_in_plus_ex_situ,
        lc_data.nhalos,
        sig=sig,
        lgL_bin_edges=lg_emline_Lbin_edges,
    )
    # take care of bins with low/zero number counts in a similar way to n_mag.get_n_data_err(), using same N_floor and N_0:
    emline_N = jnp.where(emline_N > N_FLOOR, emline_N, N_0)

    lg_emline_LF_model = jnp.log10(emline_N / lc_data.lc_vol_mpc3)

    return lg_emline_LF_model
