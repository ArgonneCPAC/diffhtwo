import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ...lightcone_generators import generate_lc_data
from ..phot_kern import get_colors_mags, mag_kern


def test_phot_kern(fake_subset_ssp_data, feniks):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    tcurves = feniks.filter_info.tcurves

    ran_key = jran.key(0)

    z_min = 0.2
    z_max = 0.5
    n_z_phot_table = 15
    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )

    num_halos = 100
    lgmp_min = 10.0
    lgmp_max = 15
    lc_sky_area_degsq = 100

    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )

    obs_mags, gal_weight, phot_kern_results = mag_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
        feniks.filter_info.mag_thresh,
        feniks.frac_cat,
    )
    assert np.isfinite(obs_mags).all()
    assert np.isfinite(gal_weight).all()
    assert np.isfinite(phot_kern_results.obs_mags).all()

    in_lh = jnp.array(list(feniks.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]
    obs_color_mag, gal_weight2, phot_kern_results2 = get_colors_mags(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
        feniks.filter_info.mag_thresh,
        in_lh_idx,
        feniks.frac_cat,
    )
    assert np.isfinite(obs_color_mag).all()
    assert np.isfinite(gal_weight2).all()
    assert np.isfinite(phot_kern_results2.obs_mags).all()
