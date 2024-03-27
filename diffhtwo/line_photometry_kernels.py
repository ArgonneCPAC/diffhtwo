"""
"""

from dsps.utils import trapz
from jax import config
from jax import jit as jjit
from jax import numpy as jnp

from .line_luminosity_kernels import LINE_LUM_NORM

config.update("jax_enable_x64", True)
L_AB = 4.4659e20  # erg/s/Hz
C_ANGSTROMS = 1e10 * 2.997e8  # angstrom/s
LINE_FLUX_NORM = LINE_LUM_NORM / C_ANGSTROMS / L_AB


@jjit
def _filter_flux_ab0_at_10pc_order_unity(wave_filter, trans_filter):
    integrand = trans_filter / wave_filter
    filter_flux_ab0 = trapz(wave_filter, integrand)
    return filter_flux_ab0


@jjit
def _ab_flux_line(line_angstrom, line_lum_cgs, tcurve_wave_aa, tcurve_trans):
    trans_at_line = jnp.interp(line_angstrom, tcurve_wave_aa, tcurve_trans)
    numerator = (line_angstrom / C_ANGSTROMS) * trans_at_line * line_lum_cgs
    filter_flux_ab0 = L_AB * _filter_flux_ab0_at_10pc_order_unity(
        tcurve_wave_aa, tcurve_trans
    )
    ab_flux = numerator / filter_flux_ab0
    return ab_flux


@jjit
def _line_ab_flux_per_mstar(
    line_wave_aa, line_trans, line_ltot_scaled, filter_flux_ab0
):
    x = line_wave_aa * line_trans * line_ltot_scaled / filter_flux_ab0
    flux_per_mstar = x * LINE_FLUX_NORM
    return flux_per_mstar
