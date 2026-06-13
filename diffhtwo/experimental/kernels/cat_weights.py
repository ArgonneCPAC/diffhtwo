import jax.numpy as jnp
from jax import jit as jjit


@jjit
def compute_cat_weights(weights, obs_mags_weighted, mag_thresh, frac_cat):
    mag_thresh = jnp.array(mag_thresh)
    mag_thresh_mask = obs_mags_weighted[:, 0] < mag_thresh[0]

    n_gals, n_bands = obs_mags_weighted.shape
    for band in range(1, n_bands):
        mag_thresh_mask *= obs_mags_weighted[:, band] < mag_thresh[band]

    return weights * jnp.where(mag_thresh_mask, frac_cat, 0.0)
