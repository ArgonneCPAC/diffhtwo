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
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad, vmap
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental.utils import safe_log10

from .n_mag import n_mag_kern, n_mag_kern_1d

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)


@jjit
def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err):
    mask = lg_n_target > -8.0
    nbins = jnp.maximum(jnp.sum(mask), 1)

    resid = lg_n_pred - lg_n_target
    chi2 = (resid / lg_n_target_err) ** 2
    chi2 = jnp.where(mask, chi2, 0.0)

    return jnp.sum(chi2) / nbins


@jjit
def get_1d_hist_from_lh_log(
    lh_centroids,
    column,
    bin_edges,
    lg_n_lh,
    lg_n_lh_err=None,
):
    """
    Project log10(number density) (and optionally its log-error)
    from Latin Hypercube samples into 1D bins.

    Parameters
    ----------
    lh_centroids : array, shape (N_samples, D)
        LH coordinates.
    column : int
        Column index to bin on.
    bin_edges : array, shape (N_bins+1,)
        Edges of the 1D histogram bins.
    lg_n_lh : array, shape (N_samples,)
        log10(number density) per LH sample (already includes n_floor).
    lg_n_lh_err : array or None
        1-sigma uncertainties in log10(number density) per LH sample.
        If None, function returns only lg_n_1d.

    Returns
    -------
    If lg_n_lh_err is None:
        lg_n_1d, bin_centers

    If lg_n_lh_err is provided:
        lg_n_1d, lg_n_1d_err, bin_centers
    """

    # 1) count LH samples in each 1D bin
    counts, _ = jnp.histogram(
        lh_centroids[:, column],
        bins=bin_edges,
    )
    counts_safe = jnp.where(counts > 0, counts, 1)

    # 2) weighted sum of log10(n)
    lg_sum, _ = jnp.histogram(
        lh_centroids[:, column],
        bins=bin_edges,
        weights=lg_n_lh,
    )

    # 3) mean log10(n) per bin
    lg_n_1d = lg_sum / counts_safe
    lg_n_1d = jnp.where(counts > 0, lg_n_1d, 0.0)

    # Return early if no errors provided
    if lg_n_lh_err is None:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return lg_n_1d, bin_centers

    # 4) If errors provided, propagate into bins:
    var_sum, _ = jnp.histogram(
        lh_centroids[:, column],
        bins=bin_edges,
        weights=lg_n_lh_err**2,
    )

    lg_n_1d_err = jnp.sqrt(var_sum) / counts_safe
    lg_n_1d_err = jnp.where(counts > 0, lg_n_1d_err, 0.0)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return lg_n_1d, lg_n_1d_err, bin_centers


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
):
    # The if structure below assumes that if len(u_theta)==1, then it is just diffstarpop params
    if len(u_theta) == 2:
        u_diffstarpop_theta, u_spspop_theta = u_theta

        u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        u_spspop_params = u_spspop_unravel(u_spspop_theta)
        spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)
    else:
        u_diffstarpop_params = u_diffstarpop_unravel(u_theta)
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        spspop_params = DEFAULT_SPSPOP_PARAMS

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

    return _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1])


loss_and_grad = jjit(value_and_grad(_loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n(
    u_theta_init,
    lg_n_target,
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
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target,
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
    lg_n_target,
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
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target,
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

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_multi_z(u_theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, grads)

    (opt_state, (loss_hist, grad_hist)) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)

    return loss_hist, grad_hist, u_theta_fit
