"""
"""

import numpy as np
from dsps.utils import triweight_gaussian

from ..line_photometry_kernels import _filter_flux_ab0_at_10pc_order_unity
from ..pred_line_photometry import emission_line_restframe_photflux_per_mstar


def test_emission_line_restframe_photflux_per_mstar():
    n_met, n_age = 6, 100

    ssp_line_ltot_scaled_table = 10 ** np.random.uniform(-4, 4, n_met * n_age)
    ssp_line_ltot_scaled_table = ssp_line_ltot_scaled_table.reshape((n_met, n_age))

    ssp_weights = np.random.uniform(0, 1, n_met * n_age).reshape((n_met, n_age))
    ssp_weights = ssp_weights / ssp_weights.sum()

    halpha = 6565.0

    line_wave_aa = halpha

    filter_wave = np.linspace(3000, 9000, 200)
    filter_trans = triweight_gaussian(filter_wave, 6250.0, 300.0) * 500
    filter_flux_ab0 = _filter_flux_ab0_at_10pc_order_unity(filter_wave, filter_trans)

    line_trans = np.interp(line_wave_aa, filter_wave, filter_trans)

    args = (
        ssp_line_ltot_scaled_table,
        ssp_weights,
        line_wave_aa,
        filter_flux_ab0,
        line_trans,
    )
    flux_per_mstar = emission_line_restframe_photflux_per_mstar(*args)
    assert flux_per_mstar.shape == ()
    assert np.isfinite(flux_per_mstar)

    mstar = 1e10
    flux_tot = mstar * flux_per_mstar
    restmag = -2.5 * np.log10(flux_tot)
    assert -30 < restmag < -10
