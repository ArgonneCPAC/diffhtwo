import jax.numpy as jnp
from jax import jit as jjit

C = 299792458.0  # copied from astropy.constants.c.value in m/s
C_ANGSTROMS = 1e10 * C  # angstrom/s


@jjit
def flux_app_from_luminosity(luminosity_cgs, redshift, cosmo):
    d_L = cosmo.luminosity_distance(redshift).to("cm").value  # Mpc to cm
    return luminosity_cgs / (4 * jnp.pi * d_L * d_L)


@jjit
def tcurve_equivalent_width(tcurve_wave_aa, tcurve_trans):
    return jnp.trapezoid(tcurve_trans, tcurve_wave_aa)


@jjit
def flux_density_aa_to_Hz(flux_density_aa, wave_obs_aa):
    return flux_density_aa * (wave_obs_aa * wave_obs_aa) / C_ANGSTROMS


@jjit
def flux_density_Hz_to_mag_AB(flux_density_Hz):
    return -2.5 * jnp.log10(flux_density_Hz) - 48.6


@jjit
def line_flux_density_aa(
    line_obs_aa,
    line_app_flux_cgs,
    tcurve_wave_aa,
    tcurve_trans,
    tcurve_equivalent_width_aa,
):
    trans_at_line = jnp.interp(line_obs_aa, tcurve_wave_aa, tcurve_trans)
    line_app_flux_band_cgs = trans_at_line * line_app_flux_cgs

    return line_app_flux_band_cgs / tcurve_equivalent_width_aa
