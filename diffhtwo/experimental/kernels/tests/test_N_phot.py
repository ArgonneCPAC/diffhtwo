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

    obs_mags_weighted, gal_cat_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks.colors[0].lc_data,
        feniks.filter_info.mag_thresh,
        feniks.frac_cat,
    )

    assert np.isfinite(obs_mags_weighted).all()
    assert np.isfinite(gal_cat_weight).all()

    # ensure that gal_weight for gals outside mag_thresh bounds is 0.0 in each band
    mag_thresh = jnp.array(feniks.filter_info.mag_thresh)
    n_gals, n_bands = obs_mags_weighted.shape
    for i in range(0, n_bands):
        mag_sel_below_faint_thresh = obs_mags_weighted[:, i] < mag_thresh[i][1] + 0.5
        gal_cat_weight_above_faint_thresh = jnp.where(
            mag_sel_below_faint_thresh, 0.0, gal_cat_weight
        )
        assert (gal_cat_weight_above_faint_thresh < 0.01).all()

        mag_sel_above_bright_thresh = obs_mags_weighted[:, i] > mag_thresh[i][0] - 0.5
        gal_cat_weight_below_bright_thresh = jnp.where(
            mag_sel_above_bright_thresh, 0.0, gal_cat_weight
        )
        assert (gal_cat_weight_below_bright_thresh < 0.01).all()
