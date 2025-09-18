from functools import partial

import jax.nn as nn
import jax.numpy as jnp
from jax import jit as jjit
from jax import random

from . import halpha_luminosity as halphaL


@partial(jjit, static_argnames=["N"])
def bimodal_SF_Q_draws(theta, N, k_SF, k_Q, dex_var=0.1):
    lgsfr_SF_draws = theta.lgsfr_SF_mean + jnp.sqrt(dex_var) * random.normal(
        k_SF, shape=(N,)
    )
    lgsfr_Q_draws = theta.lgsfr_Q_mean + jnp.sqrt(dex_var) * random.normal(
        k_Q, shape=(N,)
    )

    i = jnp.arange(N) + 0.5
    SF_weight = nn.sigmoid((N * theta.frac_SF - i))

    return lgsfr_SF_draws, lgsfr_Q_draws, SF_weight


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
    gal_lgmet_scatter=0.1,
    N=10000,
):
    lgsfr_SF_draws, lgsfr_Q_draws, SF_weights = bimodal_SF_Q_draws(theta, N, k_SF, k_Q)

    gal_t_table = jnp.linspace(0.05, 13.8, 100)  # age of the universe in Gyr

    gal_sfr_SF_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lgsfr_SF_draws
    )  # SFR in Msun/yr
    gal_sfr_SF_tables = gal_sfr_SF_tables.T

    gal_sfr_Q_tables = jnp.ones((gal_t_table.size, N)) * (
        10**lgsfr_Q_draws
    )  # SFR in Msun/yr
    gal_sfr_Q_tables = gal_sfr_Q_tables.T

    L_halpha_cgs_SF, L_halpha_unit_SF = halphaL.get_L_halpha_vmap(
        gal_sfr_SF_tables,
        gal_lgmet,
        gal_lgmet_scatter,
        gal_t_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
    )

    L_halpha_cgs_Q, L_halpha_unit_Q = halphaL.get_L_halpha_vmap(
        gal_sfr_Q_tables,
        gal_lgmet,
        gal_lgmet_scatter,
        gal_t_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
    )

    lgL_bin_edges, tw_hist_weighted_SF = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs_SF, SF_weights
    )

    _, tw_hist_weighted_Q = halphaL.get_halpha_luminosity_func(
        L_halpha_cgs_Q, 1 - SF_weights
    )

    return (
        lgL_bin_edges,
        tw_hist_weighted_SF,
        tw_hist_weighted_Q,
        SF_weights,
    )
