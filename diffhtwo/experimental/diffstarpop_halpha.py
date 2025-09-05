# flake8: noqa: E402
""" """
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random as jran
from jax import jit as jjit
from jax import vmap

import numpy as np
from collections import namedtuple
from dsps.metallicity import umzr
from diffsky.experimental.lc_phot_kern import diffstarpop_lc_cen_wrapper
from diffsky.experimental.lc_phot_kern import _calc_lgmet_weights_galpop
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table


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


def diffstarpop_halpha_kern(
    ran_key,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_luminosity,
    diffstarpop_params,
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
        t_table, diffstar_galpop.sfh_ms, ssp_lg_age_gyr, t_obs
    )

    smooth_age_weights_q = calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_lg_age_gyr, t_obs
    )

    # get metallicity weights
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    lgmet_weights_ms = _calc_lgmet_weights_galpop(
        lgmet_med_ms, LGMET_SCATTER, ssp_lgmet
    )
    lgmet_weights_q = _calc_lgmet_weights_galpop(lgmet_med_q, LGMET_SCATTER, ssp_lgmet)

    # age weights * metallicity weights
    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q

    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))

    _w_smooth_ms = ssp_weights_smooth_ms.reshape((n_gals, 1, n_met, n_age))
    _w_q = ssp_weights_q.reshape((n_gals, 1, n_met, n_age))

    halpha_L_Lsun_per_Msun_smooth_ms = jnp.sum(ssp_halpha_luminosity * _w_smooth_ms)
    halpha_L_cgs_smooth_ms = halpha_L_Lsun_per_Msun_smooth_ms * (L_SUN_CGS * _mstar_ms)

    halpha_L_Lsun_per_Msun_q = jnp.sum(ssp_halpha_luminosity * _w_q)
    halpha_L_cgs_q = halpha_L_Lsun_per_Msun_q * (L_SUN_CGS * _mstar_q)

    weights_q = diffstar_galpop.frac_q
    weights_smooth_ms = 1 - diffstar_galpop.frac_q

    halpha_L = LCLINE_EMPTY._replace(
        halpha_L_cgs_smooth_ms=halpha_L_cgs_smooth_ms,
        halpha_L_cgs_q=halpha_L_cgs_q,
        weights_smooth_ms=weights_smooth_ms,
        weights_q=weights_q,
    )

    return halpha_L
