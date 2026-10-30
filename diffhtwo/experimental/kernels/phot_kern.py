from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit

from .cat_weight import compute_mag_weight
from .lc_phot_kern import mc_phot_kern_merging_wrapper


@jjit
def mag_kern(
    ran_key,
    param_collection,
    lc_data,
    mag_thresh,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    mc_merge=0,
):
    phot_data, phot_randoms, merging_randoms = mc_phot_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    mag_weight_ms = compute_mag_weight(phot_data.obs_mags_ms, mag_thresh)
    mag_weight_q = compute_mag_weight(phot_data.obs_mags_q, mag_thresh)
    mag_weight_bursty = compute_mag_weight(phot_data.obs_mags_bursty, mag_thresh)

    return gal_weight, mag_weight_ms, mag_weight_q, mag_weight_bursty, phot_data
