import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..phot_kern import get_colors_mags, mag_kern


def test_phot_kern(feniks, feniks_lc_data):
    ran_key = jran.key(0)

    obs_mags, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
        feniks.filter_info.mag_thresh,
        feniks.frac_cat,
    )
    assert np.isfinite(obs_mags).all()
    assert np.isfinite(gal_weight).all()
    assert (gal_weight >= 0).all()
    assert np.isfinite(phot_kern_results.obs_mags).all()

    in_lh = jnp.array(list(feniks.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]
    obs_color_mag, gal_weight2, phot_kern_results2 = get_colors_mags(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        feniks_lc_data,
        feniks.filter_info.mag_thresh,
        in_lh_idx,
        feniks.frac_cat,
    )
    assert np.isfinite(obs_color_mag).all()
    assert np.isfinite(gal_weight2).all()
    assert (gal_weight2 >= 0).all()
    assert np.isfinite(phot_kern_results2.obs_mags).all()
