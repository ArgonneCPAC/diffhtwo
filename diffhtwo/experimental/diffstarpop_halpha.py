# flake8: noqa: E402
""" """
from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

import jax
import jax.numpy as jnp
from diffsky.burstpop import diffqburstpop_mono, freqburst_mono
from diffsky.dustpop import tw_dustpop_mono_noise
from diffsky.experimental.lc_phot_kern import diffstarpop_lc_cen_wrapper
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import random as jran
from jax import vmap
from jax.debug import print

from . import halpha_luminosity

LGMET_SCATTER = 0.2

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

# halpha rest wavelength center in fsps
HALPHA_CENTER_AA = 6564.5131


_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(
        diffqburstpop_mono.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B
    )
)

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)


_D = (None, None, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)


_LCLINE_RET_KEYS = (
    "halpha_L_cgs_q",
    "halpha_L_cgs_smooth_ms",
    "halpha_L_cgs_bursty_ms",
    "weights_q",
    "weights_smooth_ms",
    "weights_bursty_ms",
)
LCLine = namedtuple("LCLine", _LCLINE_RET_KEYS)
LCLINE_EMPTY = LCLine._make([None] * len(LCLine._fields))


def get_halpha_wave_eff_table(z_phot_table):
    return HALPHA_CENTER_AA / (1 + z_phot_table)


@jjit
def diffstarpop_halpha_kern(
    diffstarpop_params,
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
):
    n_met, n_age = ssp_halpha_luminosity.shape
    n_gals = logmp0.size

    ran_key, sfh_key = jran.split(ran_key, 2)
    diffstar_galpop = diffstarpop_lc_cen_wrapper(
        diffstarpop_params, sfh_key, mah_params, logmp0, t_table, t_obs
    )

    # get age weights
    smooth_age_weights_ms = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )

    smooth_age_weights_q = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    # get bursty age weights
    _args = (
        spspop_params.burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    bursty_age_weights_ms, burst_params = _calc_bursty_age_weights_vmap(*_args)

    # get p_burst_ms
    p_burst_ms = freqburst_mono.get_freqburst_from_freqburst_params(
        spspop_params.burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )

    # get metallicity weights
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    lgmet_weights_ms = _calc_lgmet_weights_galpop(
        lgmet_med_ms, LGMET_SCATTER, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = _calc_lgmet_weights_galpop(
        lgmet_med_q, LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    # age weights * metallicity weights
    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q

    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = bursty_age_weights_ms.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    wave_eff_galpop = jnp.interp(z_obs, z_phot_table, wave_eff_table)

    # get ftrans due to dust
    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    ftrans_args_q = (
        spspop_params.dustpop_params,
        HALPHA_CENTER_AA,
        diffstar_galpop.logsm_obs_q,
        diffstar_galpop.logssfr_obs_q,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args_q)
    ftrans_q = _res[1]
    print("ftrans_q.shape:{}", ftrans_q.shape)

    ftrans_args_ms = (
        spspop_params.dustpop_params,
        HALPHA_CENTER_AA,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args_ms)
    ftrans_ms = _res[1]
    print("ftrans_ms.shape:{}", ftrans_ms.shape)

    _mstar_q = 10**diffstar_galpop.logsm_obs_q
    _mstar_ms = 10**diffstar_galpop.logsm_obs_ms

    integrand_q = ssp_halpha_luminosity * ssp_weights_q
    halpha_L_cgs_q = jnp.sum(integrand_q, axis=(1, 2)) * (L_SUN_CGS * _mstar_q)

    integrand_smooth_ms = ssp_halpha_luminosity * ssp_weights_smooth_ms
    halpha_L_cgs_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(1, 2)) * (
        L_SUN_CGS * _mstar_ms
    )

    integrand_bursty_ms = ssp_halpha_luminosity * ssp_weights_bursty_ms
    halpha_L_cgs_bursty_ms = jnp.sum(integrand_bursty_ms, axis=(1, 2)) * (
        L_SUN_CGS * _mstar_ms
    )

    weights_q = diffstar_galpop.frac_q
    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - p_burst_ms)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * p_burst_ms

    halpha_L = LCLINE_EMPTY._replace(
        halpha_L_cgs_q=halpha_L_cgs_q,
        halpha_L_cgs_smooth_ms=halpha_L_cgs_smooth_ms,
        halpha_L_cgs_bursty_ms=halpha_L_cgs_bursty_ms,
        weights_q=weights_q,
        weights_smooth_ms=weights_smooth_ms,
        weights_bursty_ms=weights_bursty_ms,
    )

    return halpha_L


@jjit
def diffstarpop_halpha_lf_weighted(halpha_L_tuple):
    # get q halpha L_cgs histogram
    halpha_L_cgs_q = halpha_L_tuple.halpha_L_cgs_q
    w_q = halpha_L_tuple.weights_q
    lgL_bin_edges, tw_hist_q = halpha_luminosity.get_halpha_luminosity_func(
        halpha_L_cgs_q, w_q
    )

    # get smooth_ms halpha L_cgs histogram
    halpha_L_cgs_smooth_ms = halpha_L_tuple.halpha_L_cgs_smooth_ms
    w_smooth_ms = halpha_L_tuple.weights_smooth_ms
    _, tw_hist_smooth_ms = halpha_luminosity.get_halpha_luminosity_func(
        halpha_L_cgs_smooth_ms, w_smooth_ms
    )

    # get bursty_ms halpha L_cgs histogram
    halpha_L_cgs_bursty_ms = halpha_L_tuple.halpha_L_cgs_bursty_ms
    w_bursty_ms = halpha_L_tuple.weights_bursty_ms
    _, tw_hist_bursty_ms = halpha_luminosity.get_halpha_luminosity_func(
        halpha_L_cgs_bursty_ms, w_bursty_ms
    )

    return lgL_bin_edges, tw_hist_q, tw_hist_smooth_ms, tw_hist_bursty_ms
