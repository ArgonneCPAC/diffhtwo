import jax.numpy as jnp
from diffsky.experimental.kernels.rapid_quenching import DEFAULT_RQ_PARAMS
from diffsky.experimental.mc_diffstarpop_wrappers import _get_sfh_info_at_t_obs
from diffsky.merging.merging_kernels import compute_x_tot_from_x_in_situ
from dsps.utils import _sigmoid
from jax import jit as jjit
from jax import vmap

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

in_axes = (None, 0, None, 0, 0)
rapid_q_sfh_table = jjit(vmap(_sigmoid, in_axes=in_axes))


def get_logsfr_obs(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    mag_thresh=None,
    frac_cat=None,
):
    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        mag_thresh=mag_thresh,
        frac_cat=frac_cat,
    )
    (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
    ) = update_logsfr_obs_with_rapid_q(phot_data, lc_data)

    return (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
        gal_weight,
        lc_data.is_central,
    )


def update_logsfr_obs_with_rapid_q(phot_data, lc_data):
    sfh_table_in_situ = update_sfh_with_rapid_q(
        phot_data.sfh_table,
        lc_data.t_table,
        lc_data.t_obs,
        phot_data.p_merge,
    )

    logsm_obs_in_situ, logssfr_obs_in_situ = _get_sfh_info_at_t_obs(
        lc_data.t_table, sfh_table_in_situ, lc_data.t_obs
    )

    logsfr_obs_in_situ = logssfr_obs_in_situ + logsm_obs_in_situ
    sfr_obs_in_situ = 10**logsfr_obs_in_situ

    p_merge = phot_data.p_merge
    sat_weight = lc_data.sat_weight
    halo_indx = lc_data.halo_indx

    sfr_obs = compute_x_tot_from_x_in_situ(
        sfr_obs_in_situ, p_merge, sat_weight, halo_indx
    )
    logsfr_obs = jnp.log10(sfr_obs)
    logsm_obs = phot_data.logsm_obs

    return (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
    )


@jjit
def update_sfh_with_rapid_q(
    sfh_table,
    t_table,
    t_obs,
    p_merge,
    p_merge_x0=DEFAULT_RQ_PARAMS.rq_p_merge_x0,
    rq_tau_gyr=10**DEFAULT_RQ_PARAMS.rq_lg_age_gyr_max,
):
    logsm_obs, logssfr_obs = _get_sfh_info_at_t_obs(t_table, sfh_table, t_obs)
    logsfr_obs = logssfr_obs + logsm_obs

    t0 = t_obs - rq_tau_gyr
    k = 100
    ylo = 10**logsfr_obs
    yhi = jnp.ones_like(logsfr_obs) * 1e-8  # quench to logsfr = -8

    sfh_table_q = rapid_q_sfh_table(t_table, t0, k, ylo, yhi)
    mask = (
        t_table[None, :] < t0[:, None]
    )  # (1, n_t_table) < (n_gal, 1) -> (n_gal, n_t_table)
    sfh_table_q = jnp.where(mask, sfh_table, sfh_table_q)

    rapid_q_sfh_weight = _get_rapid_q_sfh_weight(p_merge, p_merge_x0)
    rapid_q_sfh_weight = rapid_q_sfh_weight.reshape(rapid_q_sfh_weight.size, 1)

    sfh_table_updated_with_rapid_q = (1 - rapid_q_sfh_weight) * sfh_table + (
        rapid_q_sfh_weight * sfh_table_q
    )

    return sfh_table_updated_with_rapid_q


@jjit
def _get_rapid_q_sfh_weight(p_merge, x0, k=100, ylo=0.0, yhi=1.0):
    return _sigmoid(p_merge, x0, k, ylo, yhi)
