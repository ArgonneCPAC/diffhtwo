import jax.numpy as jnp
from jax import jit as jjit


@jjit
def compute_cat_weights(weights, phot_kern_results, mag_thresh, frac_cat):
    obs_mags = phot_kern_results.obs_mags
    n_gals, n_bands = obs_mags.shape
    mag_thresh_mask = jnp.ones((n_gals,), dtype=bool)

    for band in range(0, n_bands):
        if mag_thresh[band] is not None:
            band_mag_thresh_mask = obs_mags[:, band] < mag_thresh[band]
            mag_thresh_mask *= band_mag_thresh_mask

    weights = weights * jnp.where(mag_thresh_mask, frac_cat, 0.0)

    return weights
