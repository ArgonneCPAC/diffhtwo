"""
"""

from jax import jit as jjit
from jax import numpy as jnp

from .line_photometry_kernels import _line_ab_flux_per_mstar


@jjit
def emission_line_restframe_photflux_per_mstar(
    ssp_line_ltot_scaled_table, ssp_weights, line_wave_aa, filter_flux_ab0, line_trans
):
    """Calculate restframe flux per mstar of an emission line integrated across
    a filter transmission curve

    Parameters
    ----------
    ssp_line_ltot_scaled_table : array, shape (n_met, n_age)
        Table of restframe fluxes of a metallicity-age grid of SSPs

    ssp_weights : array, shape (n_met, n_age)
        PDF weights. Assumed to sum to unity.

    line_wave_aa : float
        Wavelength of the line in angstrom

    filter_flux_ab0 : float
        Normalization term in magnitude calculation
        filter_flux_ab0 = _filter_flux_ab0_at_10pc_order_unity(wave_filter, trans_filter)

    line_trans : float
        Filter transmission curve evaluated at the wavelength of the line

    Returns
    -------
    flux_per_mstar : float
        restframe_mag = -2.5*log10(flux_per_mstar)

    """
    line_ltot_scaled = jnp.sum(ssp_line_ltot_scaled_table * ssp_weights)
    args = (line_wave_aa, line_trans, line_ltot_scaled, filter_flux_ab0)
    flux_per_mstar = _line_ab_flux_per_mstar(*args)
    return flux_per_mstar
