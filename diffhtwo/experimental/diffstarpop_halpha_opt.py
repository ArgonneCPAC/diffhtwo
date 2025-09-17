# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

from functools import partial

import jax.numpy as jnp
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from .diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha
from .diffstarpop_halpha import (
    diffstarpop_halpha_lf_weighted as dpop_halpha_lf_weighted,
)

theta0, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)
IDX = IDX = jnp.arange(8, 56, 1)


@jjit
def _mse(halpha_lf_weighted_composite_true, halpha_lf_weighted_composite_pred):
    diff = halpha_lf_weighted_composite_true - halpha_lf_weighted_composite_pred
    return jnp.mean(jnp.square(diff))


def make_subspace_loss(unravel_fn, theta_default_flat, IDX):
    """
    Build a loss that optimizes ONLY the parameters at flat indices `IDX`.
    - unravel_fn: from ravel_pytree(template)
    - theta_default_flat: 1D base vector (others stay fixed to these values)
    - IDX: 1D array/list of flat indices to vary (static for the compiled fn)

    Notes: The only thing you should not do is use namedtuple._replace() / ._asdict()
            inside @jjit. Those are Python-side and will slow/break JIT.
    """
    IDX = jnp.asarray(IDX, dtype=jnp.int32)  # capture in closure

    @jjit
    def _loss_kern_subspace(
        theta_var,  # only the selected subset: shape (len(IDX),)
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
        # scatter the subset into the full flat vector
        theta_full = theta_default_flat.at[IDX].set(theta_var)

        # back to structured params and do the usual
        diffstarpop_params = unravel_fn(theta_full)

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

    return _loss_kern_subspace


loss_kern = make_subspace_loss(unravel_fn, theta0, IDX)
loss_and_grad_fn = jjit(value_and_grad(loss_kern))


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

    def _opt_update(opt_state, i):
        theta = get_params(opt_state)
        loss, grads = loss_and_grad_fn(theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))

    theta_best_fit = get_params(opt_state)

    return loss_hist, theta_best_fit
