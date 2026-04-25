from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffhalos.lightcone_generators import mc_lightcone as mcl
from diffmah import logmh_at_t_obs
from diffmah.diffmah_kernels import _log_mah_kern
from diffsky.experimental import lightcone_generators as lcg
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.phot_utils import get_wave_eff_table
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from .lc_utils import zbin_vol, zbin_vol_vmap

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
    lh_centroids=None,
    d_centroids=None,
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

    if lh_centroids is not None:
        lh_centroids_lo = lh_centroids - (d_centroids / 2)
        lh_centroids_hi = lh_centroids + (d_centroids / 2)
        lh_vol_mpc3 = zbin_vol_vmap(
            sky_area_degsq,
            lh_centroids_lo[:, -1],
            lh_centroids_hi[:, -1],
            cosmo_params,
        )

        lc_tot_vol_mpc3 = zbin_vol(sky_area_degsq, z_min, z_max, cosmo_params)

        fields = (*lc_data._fields, "lc_tot_vol_mpc3", "sky_area_degsq", "lh_vol_mpc3")
        values = (*lc_data, lc_tot_vol_mpc3, sky_area_degsq, lh_vol_mpc3)
        lc_data = namedtuple(lc_data.__class__.__name__, fields)(*values)

    else:
        lc_tot_vol_mpc3 = zbin_vol(sky_area_degsq, z_min, z_max, cosmo_params)

        fields = (*lc_data._fields, "lc_tot_vol_mpc3", "sky_area_degsq")
        values = (*lc_data, lc_tot_vol_mpc3, sky_area_degsq)
        lc_data = namedtuple(lc_data.__class__.__name__, fields)(*values)

    return lc_data


def lc_photdata(
    ran_key,
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

    n_host_halos = halopop.logmp_obs.size
    nhalos = jnp.ones((n_host_halos,))
    nhalos_host_subs = jnp.repeat(nhalos, halopop.nsub_per_host)
    nhalos_host_all = jnp.concatenate((nhalos, nhalos_host_subs))

    n_subhalos = nhalos_host_all.size - n_host_halos
    nsubhalos = jnp.ones((n_subhalos,))

    # combine halo and subhalo weights
    nhalos = jnp.concatenate((nhalos, nsubhalos))

    # combine halo and subhalo z_obs
    z_obs_subs = jnp.repeat(halopop.z_obs, halopop.nsub_per_host)
    z_obs_all = jnp.concatenate((halopop.z_obs, z_obs_subs))
    halopop = halopop._replace(z_obs=z_obs_all)

    # combine halo and subhalo t_obs
    t_obs_subs = jnp.repeat(halopop.t_obs, halopop.nsub_per_host)
    t_obs_all = jnp.concatenate((halopop.t_obs, t_obs_subs))
    halopop = halopop._replace(t_obs=t_obs_all)

    # combine halo and subhalo logmp_obs
    subpop_logmp_obs = halopop.logmu_obs + jnp.repeat(
        halopop.logmp_obs, halopop.nsub_per_host
    )
    logmp_obs_all = jnp.concatenate((halopop.logmp_obs, subpop_logmp_obs))
    halopop = halopop._replace(logmp_obs=logmp_obs_all)
    logmp_infall = halopop.logmp_obs

    # get mah_params of subhalos from halopop.mah_params
    mah_params_names = halopop.mah_params._fields
    mah_params_sub = np.zeros((len(mah_params_names), n_subhalos))
    for i, _param in enumerate(mah_params_names):
        mah_params_sub[i, :] = halopop.mah_params._asdict()[_param][n_host_halos:]
    mah_params_sub = namedtuple("mah_params", halopop.mah_params._fields)(
        *mah_params_sub
    )

    # compute mah values at z=0 for subs and combine halo and subhalo logmp0
    logmp0_subs = _log_mah_kern(mah_params_sub, 10**halopop.logt0, halopop.logt0)
    logmp0_all = jnp.concatenate((halopop.logmp0, logmp0_subs))
    halopop = halopop._replace(logmp0=logmp0_all)

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
        nhalos,
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
