import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..cat_weight import compute_cat_weight
from ..lc_phot_kern import mc_phot_kern_merging_wrapper


def test_compute_cat_weight(feniks, feniks_lc_data):
    ran_key = jran.key(0)

    phot_kern_results = mc_phot_kern_merging_wrapper(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
    )
    obs_mags_weighted = phot_kern_results.obs_mags_weighted
    gal_weight = feniks_lc_data.cen_weight * feniks_lc_data.sat_weight
    assert np.isfinite(gal_weight).all()
    assert (gal_weight >= 0).all()

    # apply mag_thresh cuts and frac_cat
    gal_cat_weight = compute_cat_weight(
        gal_weight, obs_mags_weighted, feniks.filter_info.mag_thresh, feniks.frac_cat
    )
    assert np.isfinite(gal_cat_weight).all()
    assert (gal_cat_weight >= 0).all()

    # ensure that gal_cat_weight does not upweight compared to gal_weight
    assert gal_cat_weight.sum() <= gal_weight.sum()

    # ensure that gal_cat_weight for gals beyond mag_thresh bounds is low in each band
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
