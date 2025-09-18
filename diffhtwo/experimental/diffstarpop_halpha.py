# flake8: noqa: E402
""" """
from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

import jax
import jax.numpy as jnp
from diffsky.experimental.lc_phot_kern import (
    _calc_lgmet_weights_galpop,
    diffstarpop_lc_cen_wrapper,
)
from dsps.metallicity import umzr
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from . import halpha_luminosity

LGMET_SCATTER = 0.2

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")

_LCLINE_RET_KEYS = (
    "halpha_L_cgs_smooth_ms",
    "halpha_L_cgs_q",
    "weights_smooth_ms",
    "weights_q",
)
LCLine = namedtuple("LCLine", _LCLINE_RET_KEYS)
LCLINE_EMPTY = LCLine._make([None] * len(LCLine._fields))

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)


@jjit
def diffstarpop_halpha_kern(
    diffstarpop_params,
    ran_key,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    mzr_params,
    spspop_params,
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

    _mstar_ms = 10**diffstar_galpop.logsm_obs_ms
    _mstar_q = 10**diffstar_galpop.logsm_obs_q

    integrand_smooth_ms = ssp_halpha_luminosity * ssp_weights_smooth_ms
    halpha_L_cgs_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(1, 2)) * (
        L_SUN_CGS * _mstar_ms
    )

    integrand_q = ssp_halpha_luminosity * ssp_weights_q
    halpha_L_cgs_q = jnp.sum(integrand_q, axis=(1, 2)) * (L_SUN_CGS * _mstar_q)

    weights_q = diffstar_galpop.frac_q
    weights_smooth_ms = 1 - diffstar_galpop.frac_q

    halpha_L = LCLINE_EMPTY._replace(
        halpha_L_cgs_smooth_ms=halpha_L_cgs_smooth_ms,
        halpha_L_cgs_q=halpha_L_cgs_q,
        weights_smooth_ms=weights_smooth_ms,
        weights_q=weights_q,
    )

    return halpha_L


@jjit
def diffstarpop_halpha_lf_weighted(halpha_L_tuple):
    halpha_L_cgs_smooth_ms = halpha_L_tuple.halpha_L_cgs_smooth_ms
    weights_smooth_ms = halpha_L_tuple.weights_smooth_ms

    lgL_bin_edges, tw_hist_smooth_ms = halpha_luminosity.get_halpha_luminosity_func(
        halpha_L_cgs_smooth_ms, weights_smooth_ms
    )

    halpha_L_cgs_q = halpha_L_tuple.halpha_L_cgs_q
    weights_q = halpha_L_tuple.weights_q

    _, tw_hist_q = halpha_luminosity.get_halpha_luminosity_func(
        halpha_L_cgs_q, weights_q
    )

    return lgL_bin_edges, tw_hist_smooth_ms, tw_hist_q
