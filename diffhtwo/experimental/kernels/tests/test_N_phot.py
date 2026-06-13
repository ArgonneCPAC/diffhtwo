import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..N_phot import N_colors_mags_lh
from ..phot_kern import mag_kern


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

    obs_mags_weighted, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks.colors[0].lc_data,
        feniks.filter_info.mag_thresh,
        feniks.frac_cat,
    )

    assert np.isfinite(obs_mags_weighted).all()
    assert np.isfinite(gal_weight).all()

    # ensure that gal_weight for gals above mag_thresh is 0.0 in each band
    mag_thresh = jnp.array(feniks.filter_info.mag_thresh)
    n_gals, n_bands = obs_mags_weighted.shape
    for i in range(0, n_bands):
        mag_sel = obs_mags_weighted[:, i] < mag_thresh[i]
        gal_weight_above_magthresh = jnp.where(mag_sel, 0, gal_weight)
        assert gal_weight_above_magthresh.sum() == 0.0
