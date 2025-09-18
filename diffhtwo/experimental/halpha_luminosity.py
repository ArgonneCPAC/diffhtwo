import jax.numpy as jnp
from diffsky import diffndhist
from dsps import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import vmap

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")


@jjit
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
    logsm_table = jnp.log10(cumulative_mstar_formed(gal_t_table, gal_sfr_table))
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = jnp.power(10, logsm_obs)

    # convert luminosity [Lsun/Msun] ---> [erg/s]
    L_halpha_Lsun_per_Msun = jnp.sum(ssp_halpha_line_luminosity * weights)
    L_halpha_erg_per_sec = L_halpha_Lsun_per_Msun * (L_SUN_CGS * mstar_obs)

    return L_halpha_erg_per_sec, L_halpha_Lsun_per_Msun


get_L_halpha_vmap = jjit(
    vmap(
        get_L_halpha,
        in_axes=(0, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )
)


@jjit
def safe_log10(x):
    EPS = 1e-12
    return jnp.log(jnp.clip(x, EPS, jnp.inf)) / jnp.log(10.0)


@jjit
def get_halpha_luminosity_func(
    L_halpha_cgs, weights, sig=0.05, dlgL_bin=0.2, lgL_min=38.0, lgL_max=45.0
):
    """
    Parameters
    ----------
    L_halpha_cgs : array of shape (n,) or (n, 1)
        h-alpha Luminosities in [erg/s]

    weights : array of shape (n,)
        weights to multiply with L_halpha_cgs

    Returns
    -------
    lgL_bin_edges : array of luminosity function bin edges in log10-space
        defined using default arguments of this function.
    tw_hist_weighted:
        luminosity function - weighted histogram counts using diffsky.diffndhist
    """

    n_L = L_halpha_cgs.size
    L_halpha = L_halpha_cgs.reshape(n_L, 1)

    # mask: valid (strictly positive & finite)
    valid = jnp.isfinite(L_halpha) & (L_halpha > 0)

    # safe log10: put invalids far below the underflow bin (won't matter b/c weight=0)
    lg_floor = lgL_min - 10.0
    lgL_halpha = jnp.where(valid, safe_log10(L_halpha), lg_floor)

    # weights: zero-out invalids
    w = jnp.where(
        valid.reshape(
            n_L,
        ),
        weights.reshape(
            n_L,
        ),
        0.0,
    )

    sig_arr = jnp.zeros_like(lgL_halpha) + sig

    lgL_bin_edges = jnp.arange(lgL_min, lgL_max, dlgL_bin)
    lgL_bin_lo = lgL_bin_edges[:-1].reshape(lgL_bin_edges[:-1].size, 1)
    lgL_bin_hi = lgL_bin_edges[1:].reshape(lgL_bin_edges[1:].size, 1)

    tw_hist_weighted = diffndhist.tw_ndhist_weighted(
        lgL_halpha, sig_arr, w, lgL_bin_lo, lgL_bin_hi
    )

    return lgL_bin_edges, tw_hist_weighted
