# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from jax import jit as jjit
from jax import vmap

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

# copied from astropy.constants.c.value in m/s
C = 299792458.0

# rest UV wavelength for continuum calculation in Angstroms
UV_WAVELENGTH_AA = 1600 + 1.713
UV_FREQUENCY_HZ = C / (UV_WAVELENGTH_AA * 1e-10)

# copied from astropy import units as u: d = (10 * u.pc).to(u.cm).value
D_10pc_in_cm = 3.085677581491367e19


@jjit
def precompute_uv_luminosity(ssp_data):
    """get uv_luminosity in units of erg/s/Msun"""
    wave_lo = UV_WAVELENGTH_AA - 50.0
    wave_hi = UV_WAVELENGTH_AA + 50.0

    mask = ((ssp_data.ssp_wave >= wave_lo) & (ssp_data.ssp_wave <= wave_hi)) * 1.0
    uv_luminosity_per_hz = jnp.trapezoid(
        ssp_data.ssp_flux * mask, ssp_data.ssp_wave, axis=-1
    ) / jnp.trapezoid(mask, ssp_data.ssp_wave)

    uv_luminosity = UV_FREQUENCY_HZ * uv_luminosity_per_hz * L_SUN_CGS
    return uv_luminosity


@jjit
def append_uv_luminosity_to_ssp_data(ssp_data):
    emline_wave_dict = ssp_data.ssp_emline_wave._asdict()
    emline_wave_dict["UV"] = UV_WAVELENGTH_AA
    EmLineWave = namedtuple("EmLineWave", emline_wave_dict.keys())
    emline_wave_with_uv = EmLineWave(**emline_wave_dict)

    precomputed_uv_luminosity = precompute_uv_luminosity(ssp_data)
    emline_luminosity_with_uv = jnp.concatenate(
        [ssp_data.ssp_emline_luminosity, precomputed_uv_luminosity[..., None]], axis=-1
    )

    ssp_data = ssp_data._replace(
        ssp_emline_wave=emline_wave_with_uv,
        ssp_emline_luminosity=emline_luminosity_with_uv,
    )
    return ssp_data


def compute_M_UV_from_L_UV(L_UV, D_10pc_in_cm=D_10pc_in_cm):
    """
    Input:
        L_UV: UV Luminosity in erg/s
    """

    f_nu = L_UV / UV_FREQUENCY_HZ / (4 * np.pi * D_10pc_in_cm**2)
    M_UV = -2.5 * np.log10(f_nu) - 48.6

    return M_UV
