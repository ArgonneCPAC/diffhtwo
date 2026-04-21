from collections import namedtuple

import jax.numpy as jnp
from diffhalos.lightcone_generators import mc_lightcone as mcl
from diffmah import logmh_at_t_obs
from diffsky.experimental import lightcone_generators as lcg
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..phot_utils import get_wave_eff_table
from . import precompute_ssp_phot as psspp
from .utils import zbin_vol

N_SFH_TABLE = 100


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


def lc_photdata(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    *,
    cosmo_params=DEFAULT_COSMOLOGY,
    logmp_cutoff=11.0,
    lgmsub_min=None,
):
    """
    Monte-Carlo generation lightcone of host halos,
    and additional data needed for photometry calculations.

    This function is an adaptation of
    diffsky.experimental.lightcone_generators.weighted_lc_photdata

    Parameters
    ----------
    ran_key: jran.key
        random key

    n_host_halos : int
        Number of host halos in the weighted lightcone

    z_min, z_max : float
        min/max redshift

    lgmp_min,lgmp_max : float
        log10 of min/max halo mass in units of Msun

    sky_area_degsq: float
        sky area in deg^2

    ssp_data : namedtuple
        SSP SED templates from DSPS

    tcurves : namedtuple, length (n_bands, )
        each field stores the name of a transmission curve
        each value stores a namedtuple dsps.data_loaders.defaults.TransmissionCurve

    z_phot_table : array, shape (n_z_phot_table, )
        Redshift grid used to tabulate precomputed SSP magnitudes

    hmf_params: namedtuple, optional kwarg
        halo mass function parameters

    logmp_cutoff: float, optional kwarg
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    cosmo_params: namedtuple, optional kwarg
        cosmological parameters

    Returns
    -------
    lc_data: namedtuple
        Population of n_halos_tot halos along with data needed to compute photometry

    halopop: namedtuple
        Population of n_halos_tot halos and subhalos
            n_halos_tot = n_sub + n_host_halos
            n_sub = nsub_per_host * n_host_halos

        halopop fields:
            z_obs: ndarray of shape (n_halos_tot, )
                redshift values

            t_obs: ndarray of shape (n_halos_tot, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (n_halos_tot, )
                mah parameters

            logmp0: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                Base-10 log of z=0 age of the Universe for the input cosmology

            nhalos: ndarray of shape (n_halos_tot, )
                weight of the (sub)halo

            nhalos_host: ndarray of shape (n_halos_tot, )
                weight of the host halo
                Equal to nhalos for central halos

            nsub_per_host: int
                number of subhalos per host halo
                    n_sub = nsub_per_host * n_host_halos
                    n_halos_tot = n_sub + n_host_halos

            logmu_obs: ndarray of shape (n_halos_tot, )
                base-10 log of mu=Msub/Mhost

            halo_indx: ndarray of shape (n_halos_tot, )
                index of the associated host halo
                for central halos: halo_indx = range(n_halos_tot)

            t_table : array
                Age of the universe in Gyr at which SFH is tabulated

            ssp_data : namedtuple
                same as input

            precomputed_ssp_mag_table : array, shape (n_z_phot_table, n_bands, n_met, n_age)

            z_phot_table : array
                same as input

            wave_eff_table : array, shape (n_z_phot_table, n_bands)
                Effective wavelength of each transmission curve
                evaluated at each redshift in z_phot_table

    """
    if lgmsub_min is None:
        lgmsub_min = lgmp_min - 0.01

    args = (
        ran_key,
        lgmp_min,
        lgmsub_min,
        z_min,
        z_max,
        sky_area_degsq,
    )
    halopop = mcl.mc_lc(*args, cosmo_params=cosmo_params, logmp_cutoff=logmp_cutoff)

    n_host = halopop.logmp_obs.size
    nhalos = jnp.ones((n_host,))
    nhalos_host_subs = jnp.repeat(nhalos, halopop.nsub_per_host)
    nhalos_host_all = jnp.concatenate((nhalos, nhalos_host_subs))

    logmp_infall = halopop.logmp_obs

    n_subhalos = halopop.z_obs.size - n_host_halos
    is_central = jnp.concatenate((jnp.ones(n_host_halos), jnp.zeros(n_subhalos)))
    is_central = is_central.astype(int)
    mah_params_host = halopop.mah_params._make(
        [x[halopop.halo_indx] for x in halopop.mah_params]
    )

    n_tot = n_host_halos + n_subhalos
    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_infall = jnp.where(is_central, t0 + jnp.zeros(n_tot), halopop.mah_params.t_peak)

    logt0 = jnp.log10(t0)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    logmhost_infall = logmh_at_t_obs(mah_params_host, halopop.t_obs, logt0)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data = lcg.LCData(
        nhalos,  # undefined
        halopop.z_obs,
        halopop.t_obs,
        halopop.logmp_obs,
        halopop.mah_params,
        halopop.logmp0,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        nhalos_host_all,
        t_infall,
        logmp_infall,
        logmhost_infall,
        is_central,
        halopop.halo_indx,
    )

    lc_data = lcg.passively_add_emlines_to_lc_data(ssp_data, lc_data)
    return lc_data
