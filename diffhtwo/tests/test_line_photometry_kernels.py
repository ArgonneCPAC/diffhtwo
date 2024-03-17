"""
"""

import numpy as np

from .. import line_photometry_kernels as lpk


def test_ab_filter_flux_factor_from_precomputed():
    halpha = 6565.0
    z_obs = 0.5
    line_wave_aa_obs = halpha * (1.0 + z_obs)

    line_lum_cgs = 3.5e39
    filter_flux_ab0 = 4.45e19
    trans_at_line_wave_obs = 0.5
    args = line_wave_aa_obs, filter_flux_ab0, trans_at_line_wave_obs
    for arg in args:
        assert np.all(np.isfinite(arg))
    ab_flux_factor = lpk._ab_filter_flux_factor_from_precomputed(*args)
    assert np.all(np.isfinite(ab_flux_factor))
    assert np.all(np.isfinite(filter_flux_ab0))
    ab_flux = ab_flux_factor * line_lum_cgs
    assert np.all(np.isfinite(ab_flux))
    # return (args, ab_flux)
    # assert np.all(np.isfinite(ab_flux)), ab_flux
