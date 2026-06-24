import jax.numpy as jnp
from dsps.utils import _sigmoid
from jax import jit as jjit


@jjit
def compute_cat_weight(gal_weight, obs_mags_weighted, mag_thresh, frac_cat):
    mag_thresh = jnp.array(mag_thresh)
    mag_weight = _get_band_mag_weight(obs_mags_weighted[:, 0], mag_thresh[0])

    n_gals, n_bands = obs_mags_weighted.shape
    for band in range(1, n_bands):
        mag_weight *= _get_band_mag_weight(obs_mags_weighted[:, band], mag_thresh[band])

    return gal_weight * mag_weight * frac_cat


@jjit
def _get_band_mag_weight(mag, mag_thresh, k=50, ylo=1.0, yhi=0.0):
    mag_weight = _sigmoid(mag, mag_thresh, k, ylo, yhi)
    return mag_weight
