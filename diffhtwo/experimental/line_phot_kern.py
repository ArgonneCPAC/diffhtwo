import jax.numpy as jnp
from jax import jit as jjit
from jax import vmap

from .defaults import C_ANGSTROMS
from .utils import safe_log10


@jjit
def _flux_app_from_luminosity(luminosity_cgs, redshift, d_L):
    return luminosity_cgs / (4 * jnp.pi * d_L * d_L)


@jjit
def _tcurve_equivalent_width(tcurve_wave_aa, tcurve_trans):
    return jnp.trapezoid(tcurve_trans, tcurve_wave_aa)


@jjit
def _flux_density_aa_to_hz(flux_density_aa, wave_obs_aa):
    return flux_density_aa * (wave_obs_aa * wave_obs_aa) / C_ANGSTROMS


@jjit
def _flux_density_hz_to_mag_ab(flux_density_hz):
    valid = flux_density_hz > 0.0
    flux_density_mag_ab = jnp.where(valid, -2.5 * jnp.log10(flux_density_hz) - 48.6, 0)
    return flux_density_mag_ab


@jjit
def flux_density_filter_aa(
    line_obs_aa,
    line_app_flux_cgs,
    tcurve_wave_aa,
    tcurve_trans,
    tcurve_equivalent_width_aa,
):
    trans_at_line = jnp.interp(line_obs_aa, tcurve_wave_aa, tcurve_trans)
    line_app_flux_band_cgs = trans_at_line * line_app_flux_cgs

    return line_app_flux_band_cgs / tcurve_equivalent_width_aa


@jjit
def line_mag(line_obs_aa, line_L_cgs, redshift, d_L, tcurve_wave_aa, tcurve_trans):
    # forward model line flux
    line_flux_app_cgs = _flux_app_from_luminosity(line_L_cgs, redshift, d_L)

    # equivalent_width
    equivalent_width_aa = _tcurve_equivalent_width(tcurve_wave_aa, tcurve_trans)

    # flux_density_aa
    flux_density_filter_AA = flux_density_filter_aa(
        line_obs_aa,
        line_flux_app_cgs,
        tcurve_wave_aa,
        tcurve_trans,
        equivalent_width_aa,
    )

    # flux_density_hz
    flux_density_filter_hz = _flux_density_aa_to_hz(flux_density_filter_AA, line_obs_aa)

    # mag_ab
    mag_ab = _flux_density_hz_to_mag_ab(flux_density_filter_hz)

    return mag_ab


_M = (0, 0, 0, 0, None, None)
line_mag_vmap = jjit(
    vmap(
        line_mag,
        in_axes=_M,
    )
)
