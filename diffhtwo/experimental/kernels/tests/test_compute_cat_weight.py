import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..cat_weight import compute_mag_weight
from ..lc_phot_kern import mc_phot_kern_merging_wrapper


def test_cat_weight(feniks, feniks_lc_data):
    ran_key = jran.key(0)

    phot_data, phot_randoms, merging_randoms = mc_phot_kern_merging_wrapper(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
    )

    gal_weight = feniks_lc_data.cen_weight * feniks_lc_data.sat_weight
    assert np.isfinite(gal_weight).all()

    obs_mags_ms = phot_data.obs_mags_ms
    obs_mags_q = phot_data.obs_mags_q
    obs_mags_bursty = phot_data.obs_mags_bursty

    mag_weight_ms = compute_mag_weight(obs_mags_ms, feniks.filter_info.mag_thresh)
    mag_weight_q = compute_mag_weight(obs_mags_q, feniks.filter_info.mag_thresh)
    mag_weight_bursty = compute_mag_weight(
        obs_mags_bursty, feniks.filter_info.mag_thresh
    )

    assert np.isfinite(mag_weight_ms).all()
    assert np.isfinite(mag_weight_q).all()
    assert np.isfinite(mag_weight_bursty).all()
