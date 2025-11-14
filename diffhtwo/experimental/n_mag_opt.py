# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from functools import partial

import jax.numpy as jnp
from diffsky.param_utils.spspop_param_utils import (
    DEFAULT_SPSPOP_U_PARAMS,
    get_bounded_spspop_params_tw_dust,
)
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad
from jax.debug import print
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from .n_mag import n_mag_kern
from .utils import safe_log10

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)


@jjit
def _mse(n_pred, n_target):
    lg_n_pred = safe_log10(n_pred, EPS=1e-24)
    lg_n_target = safe_log10(n_target, EPS=1e-24)
    return jnp.mean(jnp.square(lg_n_pred - lg_n_target))


@jjit
def _mae(n_pred, n_target):
    lg_n_pred = safe_log10(n_pred, EPS=1e-24)
    lg_n_target = safe_log10(n_target, EPS=1e-24)
    return jnp.mean(jnp.abs(lg_n_pred - lg_n_target))


@jjit
def get_1d_hist_from_lh_counts(lh_centroids, column, bin_edges, n):
    n_1d, _ = jnp.histogram(lh_centroids[:, column], bins=bin_edges, weights=n)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return n_1d, bin_centers


@jjit
def _loss_kern(
    u_theta,
    n_target_1d,
    ran_key,
    lc_halopop,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    tcurves,
    lh_centroids,
    fit_column,
    fit_bin_edges,
    dmag,
):
    u_diffstarpop_theta, u_spspop_theta = u_theta

    # back to diffstarpop namedtuple u_params and then convert to bounded params
    u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
    diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

    # back to spspop namedtuple u_params and then convert to bounded params
    u_spspop_params = u_spspop_unravel(u_spspop_theta)
    spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)

    n_model = n_mag_kern(
        diffstarpop_params,
        spspop_params,
        ran_key,
        lc_halopop,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        tcurves,
        lh_centroids,
        dmag,
    )
    n_model_1d, _ = get_1d_hist_from_lh_counts(
        lh_centroids, fit_column, fit_bin_edges, n_model
    )

    return _mse(n_model_1d, n_target_1d)


loss_and_grad_fn = jjit(value_and_grad(_loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n(
    u_theta_init,
    n_target_1d,
    ran_key,
    lc_halopop,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    ssp_err_pop_params,
    tcurves,
    lh_centroids,
    fit_column,
    fit_bin_edges,
    dmag,
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        n_target_1d,
        ran_key,
        lc_halopop,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        tcurves,
        lh_centroids,
        fit_column,
        fit_bin_edges,
        dmag,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_fn(u_theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit
