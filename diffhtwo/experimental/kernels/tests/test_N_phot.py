import numpy as np
import pytest
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..N_phot import N_colors_mags_lh
from ..phot_kern import mag_kern


@pytest.mark.skip(reason="latin hypercube cube is currently not maintained")
def test_N_colors_mags_lh(feniks_single_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_single_z_data

    ran_key = jran.key(0)

    N = N_colors_mags_lh(
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
        DEFAULT_PARAM_COLLECTION,
    )

    assert np.isfinite(N).all()
    assert (N >= 0.0).all()


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
