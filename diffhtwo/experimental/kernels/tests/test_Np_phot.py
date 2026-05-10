import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..Np_phot import N_colors_mags_lh


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
