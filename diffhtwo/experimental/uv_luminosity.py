# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from collections import namedtuple

import jax.numpy as jnp
from jax import jit as jjit
from jax import vmap

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

# copied from astropy.constants.c.value in m/s
C = 299792458.0

# rest UV wavelength for continuum calculation in Angstroms
UV_WAVELENGTH_AA = 1500 + 1.713
UV_FREQUENCY_HZ = C / (UV_WAVELENGTH_AA * 1e-10)

_A = (None, None, 0)
interp_vmap = jjit(vmap(jnp.interp, in_axes=_A))

_B = (None, None, 1)
interp_vmap2 = jjit(vmap(interp_vmap, in_axes=_B))


@jjit
def precompute_uv_luminosity(ssp_data):
    """get uv_luminosity in units of erg/s/Msun"""

    uv_luminosity_per_hz = interp_vmap2(
        UV_WAVELENGTH_AA, ssp_data.ssp_wave, ssp_data.ssp_flux
    ).T

    uv_luminosity = UV_FREQUENCY_HZ * uv_luminosity_per_hz * L_SUN_CGS
    return uv_luminosity


@jjit
def append_uv_luminosity_to_ssp_data(ssp_data):
    emline_wave_dict = ssp_data.ssp_emline_wave._asdict()
    emline_wave_dict["UV"] = 1500.00
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
