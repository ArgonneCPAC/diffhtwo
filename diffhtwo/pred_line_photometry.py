"""
"""

from jax import jit as jjit
from jax import numpy as jnp

from .line_photometry_kernels import _line_ab_flux_per_mstar


@jjit
def emission_line_photflux_per_mstar(
    ssp_line_ltot_scaled_table, ssp_weights, line_wave_aa, filter_flux_ab0, line_trans
):
    line_ltot_scaled = jnp.sum(ssp_line_ltot_scaled_table * ssp_weights)
    args = (line_wave_aa, line_trans, line_ltot_scaled, filter_flux_ab0)
    flux_per_mstar = _line_ab_flux_per_mstar(*args)
    return flux_per_mstar
