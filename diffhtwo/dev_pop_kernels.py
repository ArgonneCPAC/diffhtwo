""""""

# flake8: noqa
from diffsky.burstpop import calc_bursty_age_weights_from_diffburstpop_params
from diffsky.dustpop.avpop_flex import get_av_from_avpop_params_singlegal
from diffsky.dustpop.deltapop import get_delta_from_deltapop_params
from diffsky.dustpop.funopop_simple import get_funo_from_funopop_params
from diffsky.dustpop.tw_dust import DustParams, calc_dust_frac_trans
from diffstarpop import get_bounded_diffstarpop_params
from diffstarpop.mc_diffstarpop import mc_diffstar_sfh_galpop
from diffstarpop.param_utils import get_all_diffstarpop_u_params
from dsps.constants import SFR_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from dsps.metallicity.mzr import DEFAULT_MET_PARAMS, mzr_model
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.utils import cumulative_mstar_formed
from jax import config
from jax import jit as jjit
from jax import numpy as jnp

from ..singlegal.diffdesi_singlegal_zdust import get_bounded_spspop_params_tw_dust

config.update("jax_enable_x64", True)

LGQH_MIN = 48  # log10(photon/s)
LGQH_MAX = 52  # log10(photon/s)

SANTORO22_ALPHA = -1.73
LGMET_SCATTER = 0.20


@jjit
def pred_diff_mags_singlez(
    u_params,
    diffmah_params,
    z_obs,
    t_table,
    sfh_table,
    p50_arr,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
    lgt0,
    fb,
    cosmo_params,
    ssp_data,
):
    spspop_u_params = u_params.spspop_u_params
    varied_u_params_diffstarpop = u_params.diffstarpop_u_params

    all_u_params_diffstarpop = get_all_diffstarpop_u_params(varied_u_params_diffstarpop)
    diffstarpop_u_params = all_u_params_diffstarpop

    # get bounded diffstarpop params
    diffstarpop_params = get_bounded_diffstarpop_params(diffstarpop_u_params)

    _res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        diffmah_params,
        p50_arr,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
        lgt0,
        fb,
    )
    diffstar_params_q, diffstar_params_ms, sfh_q, sfh_ms, frac_q = _res

    # get bounded spspop params
    spspop_params = get_bounded_spspop_params_tw_dust(spspop_u_params)

    args = (
        spspop_params,
        cosmo_params,
        ssp_data,
        ssp_obsmag_photflux_table,
        ssp_kcorrect_photflux_nodimming_table,
        wave_eff_aa_obsmag,
        wave_eff_aa_kcorrect,
        z_obs,
        t_table,
        sfh_table,
    )


@jjit
def calc_approx_galflux_and_lineflux(
    spspop_params,
    emlinepop_params,
    cosmo_params,
    ssp_data,
    ssp_obsmag_photflux_table,
    htwo_emlineflux_table,
    wave_eff_aa_obsmag,
    wave_emlines,
    z_obs,
    t_table,
    sfr_table,
):
    sfr_table = jnp.where(sfr_table < SFR_MIN, SFR_MIN, sfr_table)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)

    logsm_obs, logssfr_obs = _compute_tobs_properties(t_obs, t_table, sfr_table)
    lgmet = mzr_model(logsm_obs, t_obs, *DEFAULT_MET_PARAMS[:-1])

    ssp_weights = calc_ssp_weights_sfh_table_lognormal_mdf(
        t_table,
        sfr_table,
        lgmet,
        LGMET_SCATTER,
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        t_obs,
    )
    smooth_age_weights = ssp_weights.age_weights

    age_weights, burst_params = calc_bursty_age_weights_from_diffburstpop_params(
        spspop_params.burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights,
    )

    frac_trans = calc_dust_ftrans_singlegal_singlewave_from_tw_dustpop_new_params(
        spspop_params.dustpop_params,
        wave_eff_aa_obsmag,
        logsm_obs,
        logssfr_obs,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
    )

    frac_trans_emlines = (
        calc_dust_ftrans_singlegal_singlewave_from_tw_dustpop_new_params(
            emlinepop_params,
            wave_emlines,
            logsm_obs,
            logssfr_obs,
            z_obs,
            ssp_data.ssp_lg_age_gyr,
        )
    )

    gal_flux_table_nodust = ssp_obsmag_photflux_table * 10**logsm_obs
    gal_flux_table = gal_flux_table_nodust * frac_trans.reshape((1, *frac_trans.shape))

    w_age = age_weights.reshape((1, age_weights.size))
    w_age = w_age / w_age.sum()
    w_met = ssp_weights.lgmet_weights.reshape((ssp_weights.lgmet_weights.size, 1))
    w_met = w_met / w_age.sum()
    weights = w_met * w_age

    gal_flux = jnp.sum(gal_flux_table * weights, axis=(0, 1))

    return gal_flux


@jjit
def calc_dust_ftrans_singlegal_singlewave_from_tw_dustpop_new_params(
    dustpop_params, wave_aa, logsm, logssfr, redshift, ssp_lg_age_gyr
):
    av = get_av_from_avpop_params_singlegal(
        dustpop_params.avpop_params, logsm, logssfr, redshift, ssp_lg_age_gyr
    )
    delta = get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = get_funo_from_funopop_params(dustpop_params.funopop_params, logsm, logssfr)

    dust_params = DustParams(av, delta, funo)
    ftrans = calc_dust_frac_trans(wave_aa, dust_params)

    return ftrans


@jjit
def _compute_tobs_properties(t_obs, t_table, sfr_table):
    lgt_obs = jnp.log10(t_obs)

    lgt_table = jnp.log10(t_table)
    mstar_table = cumulative_mstar_formed(t_table, sfr_table)
    logsm_table = jnp.log10(mstar_table)
    logsfr_table = jnp.log10(sfr_table)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    logsfr_obs = jnp.interp(lgt_obs, lgt_table, logsfr_table)
    logssfr_obs = logsfr_obs - logsm_obs

    return logsm_obs, logssfr_obs
