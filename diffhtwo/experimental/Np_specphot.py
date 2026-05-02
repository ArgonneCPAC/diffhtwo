from functools import partial

import jax.numpy as jnp
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from . import diffndhist as diffndhist2
from .n_specphot import get_colors_mags


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def N_colors_mags_lh(
    ran_key,
    meta_data,
    fitting_data,
    param_collection,
    redshift_as_last_dimension_in_lh=True,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    obs_color_mag, weights = get_colors_mags(
        ran_key,
        param_collection,
        fitting_data.lc_data,
        meta_data.mag_columns,
        meta_data.mag_thresh_column,
        meta_data.mag_thresh,
        meta_data.frac_cat,
    )

    # calculate number density in LH bins
    sig = jnp.zeros(fitting_data.lh_centroids.shape) + (fitting_data.d_centroids / 2)
    lh_centroids_lo = fitting_data.lh_centroids - (fitting_data.d_centroids / 2)
    lh_centroids_hi = fitting_data.lh_centroids + (fitting_data.d_centroids / 2)

    if redshift_as_last_dimension_in_lh:
        z_obs = fitting_data.lc_data.z_obs.reshape(fitting_data.lc_data.z_obs.size, 1)
        obs_color_mag = jnp.hstack((obs_color_mag, z_obs))

        N = diffndhist2.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    else:
        N = diffndhist2.tw_ndhist_weighted(
            obs_color_mag,
            sig,
            weights,
            lh_centroids_lo,
            lh_centroids_hi,
        )

    return N
