# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from functools import partial

import jax.numpy as jnp
from diffsky.param_utils.spspop_param_utils import (
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
    get_bounded_spspop_params_tw_dust,
)
from diffsky.ssp_err_model.defaults import (
    ZERO_SSPERR_PARAMS,
    ZERO_SSPERR_U_PARAMS,
    get_bounded_ssperr_params,
)
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad, vmap
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental.utils import safe_log10

from .n_mag import n_mag_kern

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_zero_ssp_err_pop_theta, u_zero_ssp_err_pop_unravel = ravel_pytree(
    ZERO_SSPERR_U_PARAMS
)


@jjit
def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
    mask = lg_n_target > lg_n_thresh
    nbins = jnp.maximum(jnp.sum(mask), 1)

    resid = lg_n_pred - lg_n_target
    chi2 = (resid / lg_n_target_err) ** 2
    chi2 = jnp.where(mask, chi2, 0.0)

    return jnp.sum(chi2) / nbins


# @jjit
# def _loss_kern_1d(
#     u_theta,
#     lg_n_target_1d,
#     ran_key,
#     lc_z_obs,
#     lc_t_obs,
#     lc_mah_params,
#     lc_logmp0,
#     lc_nhalos,
#     lc_vol_mpc3,
#     t_table,
#     ssp_data,
#     precomputed_ssp_mag_table,
#     z_phot_table,
#     wave_eff_table,
#     mzr_params,
#     scatter_params,
#     ssp_err_pop_params,
#     bin_centers_1d,
#     dmag,
#     mag_column,
#     cosmo_params,
#     fb,
# ):
#     # The if structure below assumes that if len(u_theta)==1, then it is just diffstarpop params
#     if len(u_theta) == 2:
#         u_diffstarpop_theta, u_spspop_theta = u_theta

#         u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
#         diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

#         u_spspop_params = u_spspop_unravel(u_spspop_theta)
#         spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)
#     else:
#         u_diffstarpop_params = u_diffstarpop_unravel(u_theta)
#         diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

#         spspop_params = DEFAULT_SPSPOP_PARAMS

#     lg_n_model_1d = n_mag_kern_1d(
#         diffstarpop_params,
#         spspop_params,
#         ran_key,
#         lc_z_obs,
#         lc_t_obs,
#         lc_mah_params,
#         lc_logmp0,
#         lc_nhalos,
#         lc_vol_mpc3,
#         t_table,
#         ssp_data,
#         precomputed_ssp_mag_table,
#         z_phot_table,
#         wave_eff_table,
#         mzr_params,
#         scatter_params,
#         ssp_err_pop_params,
#         bin_centers_1d,
#         dmag,
#         mag_column,
#         cosmo_params,
#         fb,
#     )

#     mse_w = 0.0
#     for i in range(0, len(lg_n_model_1d)):
#         mse_w += _mse_w(lg_n_model_1d[i], lg_n_target_1d[i][0], lg_n_target_1d[i][1])

#     return mse_w


# loss_and_grad_1d = jjit(value_and_grad(_loss_kern_1d))


# @partial(jjit, static_argnames=["n_steps", "step_size"])
# def fit_n_1d(
#     u_theta_init,
#     lg_n_target_1d,
#     ran_key,
#     lc_z_obs,
#     lc_t_obs,
#     lc_mah_params,
#     lc_logmp0,
#     lc_nhalos,
#     lc_vol_mpc3,
#     t_table,
#     ssp_data,
#     precomputed_ssp_mag_table,
#     z_phot_table,
#     wave_eff_table,
#     mzr_params,
#     scatter_params,
#     ssp_err_pop_params,
#     bin_centers_1d,
#     dmag,
#     mag_column,
#     cosmo_params,
#     fb,
#     n_steps=2,
#     step_size=0.1,
# ):
#     opt_init, opt_update, get_params = jax_opt.adam(step_size)
#     opt_state = opt_init(u_theta_init)

#     other = (
#         lg_n_target_1d,
#         ran_key,
#         lc_z_obs,
#         lc_t_obs,
#         lc_mah_params,
#         lc_logmp0,
#         lc_nhalos,
#         lc_vol_mpc3,
#         t_table,
#         ssp_data,
#         precomputed_ssp_mag_table,
#         z_phot_table,
#         wave_eff_table,
#         mzr_params,
#         scatter_params,
#         ssp_err_pop_params,
#         bin_centers_1d,
#         dmag,
#         mag_column,
#         cosmo_params,
#         fb,
#     )

#     def _opt_update(opt_state, i):
#         u_theta = get_params(opt_state)
#         loss, grads = loss_and_grad_1d(u_theta, *other)
#         opt_state = opt_update(i, grads, opt_state)
#         return opt_state, (loss, grads)

#     (opt_state, (loss_hist, grad_hist)) = lax.scan(
#         _opt_update, opt_state, jnp.arange(n_steps)
#     )
#     u_theta_fit = get_params(opt_state)

#     return loss_hist, grad_hist, u_theta_fit


@jjit
def _loss_kern(
    u_theta,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_logmp0,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag,
    mag_column,
    cosmo_params,
    fb,
):
    # The if structure below assumes that if len(u_theta)==1, then it is just diffstarpop params
    if len(u_theta) == 3:
        u_diffstarpop_theta, u_spspop_theta, u_ssp_err_pop_theta = u_theta

        u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        u_spspop_params = u_spspop_unravel(u_spspop_theta)
        spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)

        u_ssp_err_pop_params = u_zero_ssp_err_pop_unravel(u_ssp_err_pop_theta)
        ssp_err_pop_params = get_bounded_ssperr_params(u_ssp_err_pop_params)

    elif len(u_theta) == 2:
        u_diffstarpop_theta, u_spspop_theta = u_theta

        u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        u_spspop_params = u_spspop_unravel(u_spspop_theta)
        spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)

        ssp_err_pop_params = ZERO_SSPERR_PARAMS

    else:
        u_diffstarpop_params = u_diffstarpop_unravel(u_theta)
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        spspop_params = DEFAULT_SPSPOP_PARAMS
        ssp_err_pop_params = ZERO_SSPERR_PARAMS

    lg_n_model, _ = n_mag_kern(
        diffstarpop_params,
        spspop_params,
        ran_key,
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        lc_logmp0,
        lc_nhalos,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        lh_centroids,
        dmag,
        mag_column,
        cosmo_params,
        fb,
    )

    return _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1], lg_n_thresh)


loss_and_grad = jjit(value_and_grad(_loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n(
    u_theta_init,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_logmp0,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag,
    mag_column,
    cosmo_params,
    fb,
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target,
        lg_n_thresh,
        ran_key,
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        lc_logmp0,
        lc_nhalos,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        lh_centroids,
        dmag,
        mag_column,
        cosmo_params,
        fb,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad(u_theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, grads)

    (opt_state, (loss_hist, grad_hist)) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)

    return loss_hist, grad_hist, u_theta_fit


_L = (
    None,
    0,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    None,
    0,
    0,
    0,
    None,
    None,
    0,
    None,
    None,
    None,
    None,
)
_loss_kern_multi_z = jjit(
    vmap(
        _loss_kern,
        in_axes=_L,
    )
)


@jjit
def _loss_total_multi_z(*args):
    return jnp.sum(_loss_kern_multi_z(*args))


loss_and_grad_multi_z = jjit(value_and_grad(_loss_total_multi_z))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n_multi_z(
    u_theta_init,
    trainable,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_logmp0,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag,
    mag_column,
    cosmo_params,
    fb,
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target,
        lg_n_thresh,
        ran_key,
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        lc_logmp0,
        lc_nhalos,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        lh_centroids,
        dmag,
        mag_column,
        cosmo_params,
        fb,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_multi_z(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, grads)

    (opt_state, (loss_hist, grad_hist)) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)

    return loss_hist, grad_hist, u_theta_fit
