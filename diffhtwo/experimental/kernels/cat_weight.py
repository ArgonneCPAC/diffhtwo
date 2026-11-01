import jax.numpy as jnp
from dsps.utils import _sigmoid
from jax import jit as jjit


@jjit
def compute_mag_weight(obs_mags_weighted, mag_thresh):
    mag_thresh = jnp.array(mag_thresh)

    n_gals, n_bands = obs_mags_weighted.shape
    mag_weight = []
    for band in range(n_bands):
        mag_bright_weight = bright_end_weight(
            obs_mags_weighted[:, band], mag_thresh[band][0]
        )
        mag_faint_weight = faint_end_weight(
            obs_mags_weighted[:, band], mag_thresh[band][1]
        )
        mag_weight_band = mag_bright_weight * mag_faint_weight
        mag_weight.append(mag_weight_band)

    mag_weight = jnp.array(mag_weight).T

    return mag_weight


@jjit
def faint_end_weight(mag, mag_thresh, k=50, ylo=1.0, yhi=0.0):
    mag_weight = _sigmoid(mag, mag_thresh, k, ylo, yhi)
    return mag_weight


@jjit
def bright_end_weight(mag, mag_thresh, k=50, ylo=0.0, yhi=1.0):
    mag_weight = _sigmoid(mag, mag_thresh, k, ylo, yhi)
    return mag_weight
