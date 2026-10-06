import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..phot_kern import get_colors_mags, mag_kern


def test_phot_kern(feniks, feniks_lc_data):
    ran_key = jran.key(0)

    obs_mags_weighted, gal_weight, mag_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
        feniks.filter_info.mag_thresh,
    )
    assert np.isfinite(obs_mags_weighted).all()

    assert np.isfinite(gal_weight).all()
    assert (gal_weight >= 0).all()

    assert np.isfinite(mag_weight).all()
    assert (mag_weight >= 0).all()

    assert np.isfinite(phot_kern_results.obs_mags).all()
    assert np.isfinite(phot_kern_results.obs_mags_weighted).all()

    obs_color_mag_weighted, gal_weight, mag_weight, phot_kern_results = get_colors_mags(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
        feniks.col_idx,
        feniks.mag_idx,
        feniks.filter_info.mag_thresh,
    )
    assert np.isfinite(obs_color_mag_weighted).all()

    assert np.isfinite(gal_weight).all()
    assert (gal_weight >= 0).all()

    assert np.isfinite(mag_weight).all()
    assert (mag_weight >= 0).all()

    assert np.isfinite(phot_kern_results.obs_mags).all()
    assert np.isfinite(phot_kern_results.obs_mags_weighted).all()
