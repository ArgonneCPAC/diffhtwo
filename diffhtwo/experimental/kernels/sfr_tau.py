from diffsky.merging.merging_kernels import get_in_plus_ex_situ_ssp_weights
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

vmap_interp = jjit(vmap(jnp.interp, in_axes=(None, None, 0)))


@jjit
def frac_stellar_mass_tau(ssp_weights, lg_age_gyr, tau_gyr):
    age_gyr = 10**lg_age_gyr

    age_weights = jnp.sum(ssp_weights, axis=1)
    cdf = jnp.cumsum(age_weights, axis=1)
    cum_weight_tau = vmap_interp(tau_gyr, age_gyr, cdf)
    return cum_weight_tau


@jjit
def compute_logsfr_tau(ssp_weights, lg_age_gyr, logsm_obs, tau_gyr=0.1):
    """
    Input:
        ssp_weights:
            SSP age-metallicity weights
            array of shape (n_gal, n_met, n_age)
        lg_age_gyr:
            SSP stellar age
            array of shape (n_age,)
        logsm_obs:
            stellar mass formed at the time of observation
            array of shape (n_gal,)
        tau_gyr: Timescale to compute SFR. Default: 100 Myr
    Returns:
        log(SFR) in a given timescale tau
            array of shape (n_gal,)
    """
    frac_sm_obs = frac_stellar_mass_tau(ssp_weights, lg_age_gyr, tau_gyr)

    sm_obs = 10**logsm_obs
    tau_yr = tau_gyr * 1e9
    sfr = (frac_sm_obs * sm_obs) / tau_yr
    return jnp.log10(sfr)


def get_logsfr_100Myr(phot_data, lc_data, ssp_data):
    mstar_in_situ = 10**phot_data.logsm_obs_in_situ
    ssp_weights_in_situ = phot_data.ssp_weights

    ssp_weights_in_plus_ex_situ = get_in_plus_ex_situ_ssp_weights(
        mstar_in_situ,
        ssp_weights_in_situ,
        phot_data.p_merge,
        lc_data.sat_weight,
        lc_data.halo_indx,
    )
    logsm_obs = phot_data.logsm_obs

    logsfr_100Myr = compute_logsfr_tau(
        ssp_weights_in_plus_ex_situ,
        ssp_data.ssp_lg_age_gyr,
        logsm_obs,
        tau_gyr=0.1,
    )
    return logsfr_100Myr
