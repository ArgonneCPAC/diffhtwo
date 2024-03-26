"""
"""

from dsps.utils import trapz
from jax import config
from jax import jit as jjit
from jax import numpy as jnp

config.update("jax_enable_x64", True)
L_AB = 4.4659e20  # erg/s/Hz
C_ANGSTROMS = 1e10 * 2.997e8


@jjit
def _filter_flux_ab0_at_10pc(wave_filter, trans_filter):
    filter_flux_ab0_by_lab = _filter_flux_ab0_at_10pc_order_unity(
        wave_filter, trans_filter
    )
    return filter_flux_ab0_by_lab * L_AB


@jjit
def _filter_flux_ab0_at_10pc_order_unity(wave_filter, trans_filter):
    integrand = trans_filter / wave_filter
    filter_flux_ab0 = trapz(wave_filter, integrand)
    return filter_flux_ab0


@jjit
def _ab_flux_line(line_angstrom, line_lum_cgs, tcurve_wave_aa, tcurve_trans):
    trans_at_line = jnp.interp(line_angstrom, tcurve_wave_aa, tcurve_trans)
    numerator = (line_angstrom / C_ANGSTROMS) * trans_at_line * line_lum_cgs
    filter_flux_ab0 = _filter_flux_ab0_at_10pc(tcurve_wave_aa, tcurve_trans)
    ab_flux = numerator / filter_flux_ab0
    return ab_flux


@jjit
def _ab_flux_line_order_unity(line_angstrom, tcurve_wave_aa, tcurve_trans):
    filter_flux_ab0_by_lab = _filter_flux_ab0_at_10pc_order_unity(
        tcurve_wave_aa, tcurve_trans
    )
    trans_at_line = jnp.interp(line_angstrom, tcurve_wave_aa, tcurve_trans)
    return trans_at_line / filter_flux_ab0_by_lab


@jjit
def _ab_filter_flux_factor_from_precomputed(
    line_wave_aa_obs, filter_flux_ab0, trans_at_line_wave_obs
):
    numerator = (line_wave_aa_obs / C_ANGSTROMS) * trans_at_line_wave_obs
    ab_flux_factor = numerator / filter_flux_ab0
    return ab_flux_factor


@jjit
def _get_precomputed_quantities(line_wave_aa, redshift, wave_filter, trans_filter):
    line_wave_aa_obs = line_wave_aa * (1.0 + redshift)
    filter_flux_ab0_order_unity = _filter_flux_ab0_at_10pc_order_unity(
        wave_filter, trans_filter
    )
