from functools import partial

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from dsps.cosmology.flat_wcdm import differential_comoving_volume_at_z
from jax import jit as jjit
from jax import vmap

from .defaults import COSMO

dV_dz = jjit(
    vmap(
        differential_comoving_volume_at_z,
        in_axes=(0, None, None, None, None),
    )
)

z_slices = 100
z_grid = jnp.linspace(0.0, 1.0, z_slices)


@jjit
def zbin_vol(sky_area_degsq, zlow, zhigh, cosmo_params):
    z = zlow + (zhigh - zlow) * z_grid

    A_sr = sky_area_degsq * (jnp.pi / 180.0) ** 2

    dV_dz_arr = dV_dz(
        z,
        cosmo_params.Om0,
        cosmo_params.w0,
        cosmo_params.wa,
        cosmo_params.h,
    )
    vol_mpc3 = jnp.trapezoid(dV_dz_arr, z) * A_sr

    return vol_mpc3


zbin_vol_vmap = jjit(vmap(zbin_vol, in_axes=(None, 0, 0, None)))


def zbin_volume(sky_area_degsq, zlow=0.2, zhigh=0.5, slices=1000):
    """
    Calculate Comoving Volume in Mpc3/h units for a given z-bin and area of survey.
    zlow: lower end of redshift bin
    zhigh: higher end of redshift bin
    slices: number of slices used for integration of dV/dz over z
    A: Survey area in deg2
    """
    z = np.linspace(zlow, zhigh, slices)
    dV_dz = np.zeros(len(z))
    A = sky_area_degsq * u.deg**2
    for i in range(0, len(z)):
        dV_dz[i] = COSMO.differential_comoving_volume(z[i]).value
    volume = (np.trapezoid(dV_dz, z) * u.Mpc**3 / u.sr) * A.to(u.sr)

    # Mpc3 units (no h dependence)
    return volume


def zbin_area(comoving_volume, zlow=0.2, zhigh=0.5, slices=1000):
    z = np.linspace(zlow, zhigh, slices)
    dV_dz = np.zeros(len(z))
    for i in range(0, len(z)):
        dV_dz[i] = COSMO.differential_comoving_volume(z[i]).value
    A_sr = (comoving_volume * u.Mpc**3) / (np.trapezoid(dV_dz, z) * u.Mpc**3 / u.sr)

    A_deg2 = A_sr.to(u.deg**2)

    # Mpc3 units (no h dependence)
    return A_deg2
