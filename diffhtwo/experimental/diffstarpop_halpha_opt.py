# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

import jax.numpy as jnp
from jax import jit as jjit
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt
from diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha
from diffstarpop_halpha import diffstarpop_halpha_lf_weighted as dpop_halpha_lf_weighted
from jax.flatten_util import ravel_pytree
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from jax import lax
from functools import partial

_, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)


@jjit
def _mse(halpha_lf_weighted_composite_true, halpha_lf_weighted_composite_pred):
    diff = halpha_lf_weighted_composite_true - halpha_lf_weighted_composite_pred
    return jnp.mean(jnp.square(diff))


def make_loss(unravel_fn):
    @jjit
    def _loss_kern(
        theta,
        halpha_lf_weighted_composite_true,
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
        diffstarpop_params = unravel_fn(theta)
        halpha_lf_pred = dpop_halpha(
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
        )

        (
            lgL_bin_edges,
            halpha_lf_weighted_smooth_ms_pred,
            halpha_lf_weighted_q_pred,
        ) = dpop_halpha_lf_weighted(halpha_lf_pred)

        halpha_lf_weighted_composite_pred = (
            halpha_lf_weighted_smooth_ms_pred + halpha_lf_weighted_q_pred
        )

        return _mse(
            halpha_lf_weighted_composite_true, halpha_lf_weighted_composite_pred
        )

    return _loss_kern


loss_kern = make_loss(unravel_fn)
loss_and_grad_func = jjit(value_and_grad(loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_diffstarpop(
    theta_init,
    halpha_lf_weighted_composite_true,
    ran_key,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    mzr_params,
    spspop_params,
    n_steps=10,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(theta_init)

    other = (
        halpha_lf_weighted_composite_true,
        ran_key,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_luminosity,
        mzr_params,
        spspop_params,
    )

    def body_fn(opt_state, i):
        theta = get_params(opt_state)
        loss, grads = loss_and_grad_func(theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(body_fn, opt_state, jnp.arange(n_steps))

    theta_best_fit = get_params(opt_state)

    return loss_hist, theta_best_fit
