import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..phot_kern import mag_kern


def test_mag_kern(feniks):
    ran_key = jran.key(0)

    gal_weight, mag_weight_ms, mag_weight_q, mag_weight_bursty, phot_data = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks.spaces[0].lc_data,
        feniks.filter_info.mag_thresh,
    )

    assert np.isfinite(gal_weight).all()

    assert np.isfinite(mag_weight_ms).all()
    assert np.isfinite(mag_weight_q).all()
    assert np.isfinite(mag_weight_bursty).all()

    assert np.isfinite(phot_data.obs_mags_ms).all()
    assert np.isfinite(phot_data.obs_mags_q).all()
    assert np.isfinite(phot_data.obs_mags_bursty).all()

    assert np.isfinite(phot_data.weights_ms).all()
    assert np.isfinite(phot_data.weights_q).all()
    assert np.isfinite(phot_data.weights_bursty).all()

    assert np.isfinite(phot_data.obs_mags).all()
    assert np.isfinite(phot_data.obs_mags_in_situ).all()
