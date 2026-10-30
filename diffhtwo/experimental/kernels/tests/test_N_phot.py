import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..phot_kern import mag_kern


def test_mag_kern(feniks):
    ran_key = jran.key(0)

    obs_mags_weighted, gal_weight, mag_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks.spaces[0].lc_data,
        feniks.filter_info.mag_thresh,
    )

    assert np.isfinite(obs_mags_weighted).all()
    assert np.isfinite(gal_weight).all()
    assert np.isfinite(mag_weight).all()
