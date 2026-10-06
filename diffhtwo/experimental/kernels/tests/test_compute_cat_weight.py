import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..cat_weight import compute_mag_weight
from ..lc_phot_kern import mc_phot_kern_merging_wrapper


def test_cat_weight(feniks, feniks_lc_data):
    ran_key = jran.key(0)

    phot_kern_results = mc_phot_kern_merging_wrapper(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
    )
    obs_mags_weighted = phot_kern_results.obs_mags_weighted
    gal_weight = feniks_lc_data.cen_weight * feniks_lc_data.sat_weight
    assert np.isfinite(gal_weight).all()

    mag_weight = compute_mag_weight(obs_mags_weighted, feniks.filter_info.mag_thresh)
    assert np.isfinite(mag_weight).all()
