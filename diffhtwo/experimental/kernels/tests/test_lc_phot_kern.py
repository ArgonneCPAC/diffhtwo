import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    DEFAULT_PARAM_COLLECTION,
    check_param_collection_is_ok,
)
from jax import random as jran

from ..lc_phot_kern import mc_phot_kern_merging_wrapper, multiband_lc_phot_kern


def test_lc_phot_kern(fake_subset_ssp_data, feniks_tcurves):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    ran_key = jran.key(0)
    z_min = 0.2
    z_max = 0.5
    num_halos = 100

    assert check_param_collection_is_ok(DEFAULT_PARAM_COLLECTION)
    lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        feniks_tcurves,
    )
    assert np.isfinite(lc_data.cen_weight).all()
    assert np.isfinite(lc_data.sat_weight).all()
    assert np.isfinite(gal_weight).all()
    assert np.isfinite(phot_kern_results.obs_mags).all()

    phot_kern_results2 = mc_phot_kern_merging_wrapper(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
    )
    assert np.isfinite(phot_kern_results2.obs_mags).all()
