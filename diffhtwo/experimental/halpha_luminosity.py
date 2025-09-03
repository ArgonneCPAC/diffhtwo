from dsps import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table
from dsps.constants import SFR_MIN
from diffsky import diffndhist

from jax import jit as jjit
from jax import vmap
import jax.numpy as jnp

from astropy.constants import L_sun

L_SUN_CGS = jnp.array(L_sun.cgs.value)


def get_L_halpha(
    gal_sfr_table,
    gal_lgmet,
    gal_lgmet_scatter,
    gal_t_table,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
):
    weights, lgmet_weights, age_weights = calc_ssp_weights_sfh_table_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs[0],
    )
    # get mass
    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, SFR_MIN)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = jnp.power(10, logsm_obs)

    # convert luminosity [Lsun/Msun] ---> [erg/s]
    L_halpha_Lsun_per_Msun = (ssp_halpha_line_luminosity * weights).sum()
    L_halpha_erg_per_sec = L_halpha_Lsun_per_Msun * (L_SUN_CGS * mstar_obs)

    return L_halpha_erg_per_sec, L_halpha_Lsun_per_Msun


get_L_halpha_vmap = jjit(
    vmap(
        get_L_halpha,
        in_axes=(0, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )
)


def get_halpha_luminosity_func(
    L_halpha_cgs, weights, sig=0.001, dlgL_bin=0.2, lgL_min=40.0, lgL_max=45.0
):
    lg_L_halpha_cgs = jnp.log10(L_halpha_cgs)

    sig = jnp.zeros_like(lg_L_halpha_cgs) + sig

    lgL_bin_edges = jnp.arange(lgL_min, lgL_max, dlgL_bin)
    lgL_bin_lo = lgL_bin_edges[:-1].reshape(lgL_bin_edges[:-1].size, 1)
    lgL_bin_hi = lgL_bin_edges[1:].reshape(lgL_bin_edges[1:].size, 1)

    tw_hist_weighted = diffndhist.tw_ndhist_weighted(
        lg_L_halpha_cgs, sig, weights, lgL_bin_lo, lgL_bin_hi
    )

    return lgL_bin_edges, tw_hist_weighted
