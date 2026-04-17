import jax.numpy as jnp
from jax import jit as jjit


@jjit
def get_ssp_emline_luminosity(emline_wave_aa, ssp_data):
    ssp_emline_wave = jnp.array(ssp_data.ssp_emline_wave)
    idx = jnp.argmin(jnp.abs(ssp_emline_wave - emline_wave_aa))
    ssp_emline_luminosity = ssp_data.ssp_emline_luminosity[:, :, idx]
    return ssp_emline_luminosity
