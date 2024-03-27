"""
"""

import numpy as np
from dsps.utils import triweight_gaussian

from .. import line_photometry_kernels as lpk


def test_ab_flux_line():
    halpha_wave_aa_obs = 6565.0
    line_lum_cgs = 3.5e39

    tcurve_wave = np.linspace(3_000, 7_000, 1_000)
    tcurve_trans = triweight_gaussian(tcurve_wave, 6250.0, 300.0) * 500
    args = (halpha_wave_aa_obs, line_lum_cgs, tcurve_wave, tcurve_trans)
    ab_line_flux = lpk._ab_flux_line(*args)
    assert ab_line_flux.shape == ()
    assert np.all(np.isfinite(ab_line_flux))

    t_line = np.interp(halpha_wave_aa_obs, tcurve_wave, tcurve_trans)
    t_ab = lpk._filter_flux_ab0_at_10pc_order_unity(tcurve_wave, tcurve_trans)

    factor = (halpha_wave_aa_obs * line_lum_cgs) / (lpk.L_AB * lpk.C_ANGSTROMS)
    x = factor * t_line / t_ab
    assert np.allclose(x, ab_line_flux)
