from collections import namedtuple
from difflib import get_close_matches

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from diffsky.experimental import lightcone_generators as lcg
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import differential_comoving_volume_at_z
from jax import jit as jjit
from jax import random as jran
from jax import vmap
from jax.tree_util import tree_flatten_with_path

from .defaults import COSMO


@jjit
def get_ssp_emline_luminosity(emline_wave_aa, ssp_data):
    ssp_emline_wave = jnp.array(ssp_data.ssp_emline_wave)
    idx = jnp.argmin(jnp.abs(ssp_emline_wave - emline_wave_aa))
    ssp_emline_luminosity = ssp_data.ssp_emline_luminosity[:, :, idx]
    return ssp_emline_luminosity


@jjit
def lupton_log10(t, log10_clip, t0=0.0, M0=0.0, alpha=1 / jnp.log(10.0)):
    """Clipped base-10 log function taken from
    https://github.com/ArgonneCPAC/shamnet/blob/d47c842bfc5ad751ad63d0b21100db709de52e58/shamnet/utils.py#L217C5-L217C17

    Parameters
    ----------
    t : ndarray of shape (n, )

    log10_clip : float
        Returned values of t larger than log10_clip will agree with log10(t).
        Values smaller than log10(t) will converge to 10**log10_clip.

    Returns
    -------
    lup : ndarray of shape (n, )

    """
    k = 10.0**log10_clip
    return M0 + alpha * (jnp.arcsinh((t - t0) / (2 * k)) + jnp.log(k))


@jjit
def safe_log10(x, EPS=1e-12):
    return jnp.log(jnp.clip(x, EPS, jnp.inf)) / jnp.log(10.0)


@jjit
def get_subset_lh(ran_key, lh_centroids, d_centroids, lg_n_data_err_lh, n_centroids):
    indices = jnp.indices((lh_centroids.shape[0],))
    indices = indices.reshape(
        indices.size,
    )
    lh_idx = jran.choice(ran_key, indices, shape=(n_centroids,), replace=False)

    lh_centroid_subset = lh_centroids[lh_idx]
    d_centroids_subset = d_centroids[lh_idx]
    lg_n_data_err_lh_subset = lg_n_data_err_lh[:, lh_idx]

    return lh_centroid_subset, d_centroids_subset, lg_n_data_err_lh_subset


def generate_lc_data(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    lc_args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_photdata(*lc_args, cosmo_params=cosmo_params)

    fields = (*lc_data._fields, "lc_vol_mpc3")
    lc_vol_mpc3 = zbin_vol(sky_area_degsq, z_min, z_max, cosmo_params)
    values = (*lc_data, lc_vol_mpc3)
    lc_data = namedtuple(lc_data.__class__.__name__, fields)(*values)

    return lc_data


dV_dz = jjit(
    vmap(
        differential_comoving_volume_at_z,
        in_axes=(0, None, None, None, None),
    )
)


@jjit
def zbin_vol(sky_area_degsq, zlow, zhigh, cosmo_params, n_slice=1000):
    z = jnp.linspace(zlow, zhigh, n_slice)
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


def get_tcurve(get_filter_number, filter_info_filename, tcurves_filename):
    with open(filter_info_filename) as INFO:
        info = INFO.readlines()
    with open(tcurves_filename) as TCURVES:
        tcurves = TCURVES.readlines()

    f_idx = get_filter_number - 1
    t_idx = tcurves.index(get_close_matches(info[f_idx], tcurves)[0])

    i = 0
    wave_aa = []
    trans = []
    while (len(tcurves[t_idx + 1 :][i].split()) <= 3) & (
        (t_idx + 2 + i) < len(tcurves)
    ):
        wave_aa.append(float(tcurves[t_idx + 1 :][i].split()[-2]))
        trans.append(float(tcurves[t_idx + 1 :][i].split()[-1]))
        i += 1

    return jnp.array(wave_aa), jnp.array(trans)


def get_feniks_filter_number_from_translate_file(translate_file, filter_name):
    idx = translate_file["col1"] == "fcol_" + filter_name
    filter_number = int(translate_file[idx]["col2"].data[0][1:])
    return filter_number


def get_param_names(params):
    paths, leaves = tree_flatten_with_path(params)
    names = [p[0][0].name for p in paths]  # each p is (GetAttrKey(...),)
    return names
