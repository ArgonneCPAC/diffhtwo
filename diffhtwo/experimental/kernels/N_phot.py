from functools import partial

import jax.numpy as jnp
from diffsky import diffndhist_lomem
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from .phot_kern import get_colors_mags, mag_kern


@jjit
def N_mags_1d(
    ran_key,
    param_collection,
    magbin_bands,
    lc_data,
    mag_thresh,
    frac_cat,
    sig_scale=0.5,
):
    obs_mags, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_thresh,
        frac_cat,
    )

    n_gals, n_bands = obs_mags.shape
    N_bands = []
    for band in range(0, n_bands):
        mags = obs_mags[:, band].reshape(obs_mags[:, band].size, 1)

        magbin_edges = magbin_bands[band]

        sig = jnp.diff(magbin_edges) * sig_scale
        sig = sig.reshape(sig.size, 1)

        mag_lo = magbin_edges[:-1].reshape(magbin_edges[:-1].size, 1)
        mag_hi = magbin_edges[1:].reshape(magbin_edges[1:].size, 1)

        N_mags = diffndhist_lomem.tw_ndhist_weighted(
            mags,
            sig,
            gal_weight,
            mag_lo,
            mag_hi,
        )
        N_bands.append(N_mags)

    return N_bands


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def N_colors_mags_lh(
    ran_key,
    meta_data,
    fitting_data,
    param_collection,
    redshift_as_last_dimension_in_lh=True,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    obs_color_mag, weights, phot_kern_results = get_colors_mags(
        ran_key,
        param_collection,
        fitting_data.lc_data,
        meta_data.mag_thresh,
        meta_data.in_lh_idx,
        meta_data.frac_cat,
    )

    # calculate number density in LH bins
    sig = jnp.zeros(fitting_data.lh_centroids.shape) + (fitting_data.d_centroids / 2)
    lh_centroids_lo = fitting_data.lh_centroids - (fitting_data.d_centroids / 2)
    lh_centroids_hi = fitting_data.lh_centroids + (fitting_data.d_centroids / 2)

    if redshift_as_last_dimension_in_lh:
        z_obs = fitting_data.lc_data.z_obs.reshape(fitting_data.lc_data.z_obs.size, 1)
        obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

        N = diffndhist_lomem.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    else:
        N = diffndhist_lomem.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    return N
