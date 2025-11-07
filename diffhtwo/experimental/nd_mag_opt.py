from functools import partial

import jax.numpy as jnp
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from .nd_mag import nd_mag_kern

u_theta_default, u_unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_U_PARAMS)


@jjit
def _mse(nd_pred, nd_target):
    return jnp.mean(jnp.square(nd_pred - nd_target))


@jjit
def _loss_kern(
    u_diffstarpop_theta,
    nd_target,
    ran_key,
    lc_halopop,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    tcurves,
    lh_centroids,
):
    # back to structured params and do the usual
    u_diffstarpop_params = u_unravel_fn(u_diffstarpop_theta)

    # convert to bounded params
    diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

    nd_model = nd_mag_kern(
        diffstarpop_params,
        ran_key,
        lc_halopop,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        tcurves,
        lh_centroids,
    )

    return _mse(nd_model, nd_target)


loss_and_grad_fn = jjit(value_and_grad(_loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_nd(
    u_diffstarpop_theta_init,
    nd_target,
    ran_key,
    lc_halopop,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    tcurves,
    lh_centroids,
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_diffstarpop_theta_init)

    other = (
        nd_target,
        ran_key,
        lc_halopop,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        tcurves,
        lh_centroids,
    )

    def _opt_update(opt_state, i):
        u_diffstarpop_theta = get_params(opt_state)
        loss, grads = loss_and_grad_fn(u_diffstarpop_theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_diffstarpop_theta_fit = get_params(opt_state)

    # back to structured params and do the usual
    u_diffstarpop_params_fit = u_unravel_fn(u_diffstarpop_theta_fit)

    # convert to bounded params
    diffstarpop_params_fit = get_bounded_diffstarpop_params(u_diffstarpop_params_fit)

    return loss_hist, diffstarpop_params_fit
