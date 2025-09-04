import halpha_luminosity as halphaL
import jax.numpy as jnp
from jax import random
import jax.nn as nn
from jax import jit as jjit
from functools import partial

# copied from astropy.constants.L_sun.cgs.value
L_SUN_CGS = jnp.array(3.828e33, dtype="float64")


def pop_model(
    theta,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
    lg_sfr_var=0.1,
    N=10000,
    gal_lgmet=-1.0,
    gal_lgmet_scatter=0,
):
    lg_sfr_mean = theta["lg_sfr_mean"]
    key = random.PRNGKey(1000)
    lg_sfr_draws = lg_sfr_mean + jnp.sqrt(lg_sfr_var) * random.normal(key, shape=(N,))

    gal_t_table = jnp.linspace(0.05, 13.8, 100)  # age of the universe in Gyr
    gal_sfr_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lg_sfr_draws
    )  # SFR in Msun/yr
    gal_sfr_tables = gal_sfr_tables.T

    L_halpha_cgs, L_halpha_unit = halphaL.get_L_halpha_vmap(
        gal_sfr_tables,
        gal_lgmet,
        gal_lgmet_scatter,
        gal_t_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
    )
    weights = jnp.ones_like(L_halpha_cgs)
    lgL_bin_edges, tw_hist_weighted = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs, weights
    )

    return lgL_bin_edges, tw_hist_weighted, L_halpha_cgs


@partial(jjit, static_argnames=["N"])
def bimodal_SF_Q_draws(theta, N, k_SF, k_Q, dex_var=0.1):
    lgsfr_SF_mean = theta["lgsfr_SF_mean"]
    frac_SF = theta["frac_SF"]
    lgsfr_Q_mean = theta["lgsfr_Q_mean"]

    lgsfr_SF_draws = lgsfr_SF_mean + jnp.sqrt(dex_var) * random.normal(k_SF, shape=(N,))
    lgsfr_Q_draws = lgsfr_Q_mean + jnp.sqrt(dex_var) * random.normal(k_Q, shape=(N,))

    i = jnp.arange(N) + 0.5
    SF_weight = nn.sigmoid((N * frac_SF - i))

    lgsfr_draws = SF_weight * lgsfr_SF_draws + (1.0 - SF_weight) * lgsfr_Q_draws

    return lgsfr_draws, SF_weight


@jjit
def pop_bimodal(
    theta,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
    k_SF,
    k_Q,
    gal_lgmet=-1.0,
    gal_lgmet_scatter=0,
    N=10000,
):
    lg_sfr_draws, SF_weights = bimodal_SF_Q_draws(theta, N, k_SF, k_Q)

    gal_t_table = jnp.linspace(0.05, 13.8, 100)  # age of the universe in Gyr
    gal_sfr_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lg_sfr_draws
    )  # SFR in Msun/yr
    gal_sfr_tables = gal_sfr_tables.T

    L_halpha_cgs, L_halpha_unit = halphaL.get_L_halpha_vmap(
        gal_sfr_tables,
        gal_lgmet,
        gal_lgmet_scatter,
        gal_t_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
    )

    Q_weights = 1 - SF_weights

    lgL_bin_edges, tw_hist_weighted_SF = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs, SF_weights
    )
    _, tw_hist_weighted_Q = halphaL.get_halpha_luminosity_func(L_halpha_cgs, Q_weights)

    return (
        lgL_bin_edges,
        L_halpha_cgs,
        tw_hist_weighted_SF,
        tw_hist_weighted_Q,
        SF_weights,
    )
