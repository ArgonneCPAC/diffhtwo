# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from functools import partial

import jax.numpy as jnp
from diffmah.defaults import DiffmahParams
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.mass_functions import mc_hosts
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
from jax import lax
from jax import random as jran
from jax import value_and_grad, vmap
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental.utils import safe_log10

from . import diffstarpop_halpha as dpop_halpha
from . import emline_luminosity
from .n_mag import N_0, N_FLOOR, n_mag_kern, n_mag_kern_1d, n_mag_kern_nocolor

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_zero_ssp_err_pop_theta, u_zero_ssp_err_pop_unravel = ravel_pytree(
    ZERO_SSPERR_U_PARAMS
)

HALPHA_WAVE_AA = 6565.09893918  # halpha_line_center_c3k

# used for fisher analysis
# @jjit
# def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
#     mask = lg_n_target > lg_n_thresh

#     resid = (lg_n_pred - lg_n_target) ** 2
#     chi2 = resid / (lg_n_target_err**2)
#     chi2 = jnp.where(mask, chi2, 0.0)

#     return jnp.sum(chi2)


@jjit
def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
    mask = lg_n_target > lg_n_thresh
    nbins = jnp.maximum(jnp.sum(mask), 1)

    resid = (lg_n_pred - lg_n_target) ** 2
    chi2 = resid / lg_n_target_err
    chi2 = jnp.where(mask, chi2, 0.0)

    return jnp.sum(chi2) / nbins


# @jjit
# def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
#     mask = lg_n_target > lg_n_thresh

#     num = jnp.sum(
#         jnp.square(lg_n_pred - lg_n_target) / jnp.square(lg_n_target_err), where=mask
#     )
#     den = jnp.sum(1 / jnp.square(lg_n_target_err), where=mask)

#     return num / den


# @jjit
# def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
#     mask = lg_n_target > lg_n_thresh

#     resid = jnp.square(lg_n_pred - lg_n_target)
#     return jnp.mean(resid, where=mask)


# @jjit
# def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
#     mask = lg_n_target > lg_n_thresh
#     nbins = jnp.maximum(jnp.sum(mask), 1)

#     resid = (lg_n_pred - lg_n_target) ** 2
#     chi2 = resid / (lg_n_target_err**2)
#     chi2 = jnp.where(mask, chi2, 0.0)

#     return jnp.sum(chi2)


# @jjit
# def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
#     mask = lg_n_target > lg_n_thresh
#     nbins = jnp.maximum(jnp.sum(mask), 1)

#     resid = lg_n_pred - lg_n_target
#     chi2 = resid**2
#     chi2 = jnp.where(mask, chi2, 0.0)

#     return jnp.mean(chi2)


# 1-D histogram bins based fitting
@jjit
def _loss_kern_1d(
    u_theta,
    lg_n_target_1d,
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
    bin_centers_1d,
    dmag,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    fit_columns,
    cosmo_params,
    fb,
    frac_cat=1.0,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_LF_z=None,
    halpha_LF_delta_z=None,
    halpha_LF_delta_z_vol_Mpc3=None,
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

    lg_n_model_1d = n_mag_kern_1d(
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
        bin_centers_1d,
        dmag,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
    )

    loss = 0.0
    for i in range(0, len(fit_columns)):
        fit_column = fit_columns[i]
        loss += _mse_w(
            lg_n_model_1d[fit_column][0],
            lg_n_target_1d[fit_column][0],
            lg_n_target_1d[fit_column][1],
            lg_n_thresh,
        )

    if lg_halpha_LF_target is not None:
        halpha_args = (
            diffstarpop_params,
            ran_key,
            lc_z_obs,
            lc_t_obs,
            lc_mah_params,
            lc_logmp0,
            t_table,
            ssp_data,
            ssp_halpha_luminosity,
            z_phot_table,
            mzr_params,
            spspop_params,
            scatter_params,
            cosmo_params,
            fb,
        )
        halpha_L = dpop_halpha.diffstarpop_halpha_kern(*halpha_args)

        halpha_LF_zmin = halpha_LF_z - (halpha_LF_delta_z / 2)
        halpha_LF_zmax = halpha_LF_z + (halpha_LF_delta_z / 2)
        halpha_LF_z_sel = (lc_z_obs > halpha_LF_zmin) & (lc_z_obs < halpha_LF_zmax)
        halpha_LF_z_sel = jnp.float64(halpha_LF_z_sel)

        (
            _,
            halpha_lf_weighted_q,
            halpha_lf_weighted_smooth_ms,
            halpha_lf_weighted_bursty_ms,
        ) = dpop_halpha.diffstarpop_halpha_lf_weighted_lc_weighted(
            halpha_L,
            lc_nhalos * halpha_LF_z_sel,
            sig=0.05,
            lgL_bin_edges=lg_halpha_Lbin_edges,
        )

        halpha_lf_weighted_composite = (
            halpha_lf_weighted_q
            + halpha_lf_weighted_smooth_ms
            + halpha_lf_weighted_bursty_ms
        )
        # take care of bins with low/zero number counts in a similar way to n_mag.get_n_data_err(), using same N_floor and N_0:
        halpha_lf_weighted_composite = jnp.where(
            halpha_lf_weighted_composite > N_FLOOR, halpha_lf_weighted_composite, N_0
        )

        lg_halpha_LF_model = jnp.log10(
            halpha_lf_weighted_composite / halpha_LF_delta_z_vol_Mpc3
        )

        loss += _mse_w(
            lg_halpha_LF_model,
            lg_halpha_LF_target[0],
            lg_halpha_LF_target[1],
            lg_n_thresh,
        )

    return loss


loss_and_grad_1d = jjit(value_and_grad(_loss_kern_1d))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n_1d(
    u_theta_init,
    lg_n_target_1d,
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
    bin_centers_1d,
    dmag,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    fit_columns,
    cosmo_params,
    fb,
    frac_cat=1.0,
    n_steps=2,
    step_size=0.1,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_LF_z=None,
    halpha_LF_delta_z=None,
    halpha_LF_delta_z_vol_Mpc3=None,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target_1d,
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
        bin_centers_1d,
        dmag,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        fit_columns,
        cosmo_params,
        fb,
        frac_cat,
        ssp_halpha_luminosity,
        lg_halpha_LF_target,
        lg_halpha_Lbin_edges,
        halpha_LF_z,
        halpha_LF_delta_z,
        halpha_LF_delta_z_vol_Mpc3,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_1d(u_theta, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, grads)

    (opt_state, (loss_hist, grad_hist)) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)

    return loss_hist, grad_hist, u_theta_fit


_L_1d = (
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
    None,
    None,
    0,
    0,
    0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
)
_loss_kern_1d_multi_z = jjit(
    vmap(
        _loss_kern_1d,
        in_axes=_L_1d,
    )
)


@jjit
def _loss_1d_total_multi_z(*args):
    return jnp.sum(_loss_kern_1d_multi_z(*args))


loss_and_grad_1d_multi_z = jjit(value_and_grad(_loss_1d_total_multi_z))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n_1d_multi_z(
    u_theta_init,
    trainable,
    lg_n_target_1d,
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
    bin_centers_1d,
    dmag,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    fit_columns,
    cosmo_params,
    fb,
    frac_cat=1.0,
    n_steps=2,
    step_size=0.1,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_LF_z=None,
    halpha_LF_delta_z=None,
    halpha_LF_delta_z_vol_Mpc3=None,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target_1d,
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
        bin_centers_1d,
        dmag,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        fit_columns,
        cosmo_params,
        fb,
        frac_cat,
        ssp_halpha_luminosity,
        lg_halpha_LF_target,
        lg_halpha_Lbin_edges,
        halpha_LF_z,
        halpha_LF_delta_z,
        halpha_LF_delta_z_vol_Mpc3,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_1d_multi_z(u_theta, *other)
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


@jjit
def get_halpha_loss(
    diffstarpop_params,
    ran_key,
    lg_halpha_LF_target,
    lg_halpha_Lbin_edges,
    lg_n_thresh,
    lc_z_obs,
    lc_t_obs,
    lc_mah_params,
    lc_nhalos,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    mzr_params,
    spspop_params,
    scatter_params,
    cosmo_params,
    fb,
):
    L_halpha_cgs, _ = emline_luminosity.compute_emline_luminosity(
        ran_key,
        lc_z_obs,
        lc_t_obs,
        lc_mah_params,
        diffstarpop_params,
        spspop_params,
        mzr_params,
        scatter_params,
        t_table,
        ssp_data,
        HALPHA_WAVE_AA,
        ssp_halpha_luminosity,
        cosmo_params,
        fb,
    )

    sig = jnp.diff(lg_halpha_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    _, halpha_N = emline_luminosity.get_emline_luminosity_func(
        L_halpha_cgs, lc_nhalos, sig=sig, lgL_bin_edges=lg_halpha_Lbin_edges
    )
    # take care of bins with low/zero number counts in a similar way to n_mag.get_n_data_err(), using same N_floor and N_0:
    halpha_N = jnp.where(halpha_N > N_FLOOR, halpha_N, N_0)

    lg_halpha_LF_model = jnp.log10(halpha_N / lc_vol_mpc3)

    return _mse_w(
        lg_halpha_LF_model,
        lg_halpha_LF_target[0],
        lg_halpha_LF_target[1],
        lg_n_thresh,
    )


# Latin Hypercube bins based fitting
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
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params,
    fb,
    frac_cat=1.0,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_lc_z_obs=None,
    halpha_lc_t_obs=None,
    halpha_lc_mah_params=None,
    halpha_lc_nhalos=None,
    halpha_lc_vol_mpc3=None,
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
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
    )
    loss = _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1], lg_n_thresh)

    if lg_halpha_LF_target is not None:
        halpha_lc_mah_params = DiffmahParams(*halpha_lc_mah_params)
        halpha_loss_args = (
            diffstarpop_params,
            ran_key,
            lg_halpha_LF_target,
            lg_halpha_Lbin_edges,
            lg_n_thresh,
            halpha_lc_z_obs,
            halpha_lc_t_obs,
            halpha_lc_mah_params,
            halpha_lc_nhalos,
            halpha_lc_vol_mpc3,
            t_table,
            ssp_data,
            ssp_halpha_luminosity,
            mzr_params,
            spspop_params,
            scatter_params,
            cosmo_params,
            fb,
        )
        loss += get_halpha_loss(*halpha_loss_args)

    return loss


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
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params,
    fb,
    frac_cat=1.0,
    n_steps=2,
    step_size=0.1,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_lc_z_obs=None,
    halpha_lc_t_obs=None,
    halpha_lc_mah_params=None,
    halpha_lc_nhalos=None,
    halpha_lc_vol_mpc3=None,
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
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
        ssp_halpha_luminosity,
        lg_halpha_LF_target,
        lg_halpha_Lbin_edges,
        halpha_lc_z_obs,
        halpha_lc_t_obs,
        halpha_lc_mah_params,
        halpha_lc_nhalos,
        halpha_lc_vol_mpc3,
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
    None,
    None,
    0,
    0,
    0,
    None,
    None,
    0,
    0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
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
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params,
    fb,
    frac_cat=1.0,
    n_steps=2,
    step_size=0.1,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_lc_z_obs=None,
    halpha_lc_t_obs=None,
    halpha_lc_mah_params=None,
    halpha_lc_nhalos=None,
    halpha_lc_vol_mpc3=None,
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
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
        ssp_halpha_luminosity,
        lg_halpha_LF_target,
        lg_halpha_Lbin_edges,
        halpha_lc_z_obs,
        halpha_lc_t_obs,
        halpha_lc_mah_params,
        halpha_lc_nhalos,
        halpha_lc_vol_mpc3,
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


@jjit
def compute_nb_loss(nb_args, nb_idx):
    (
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
        nb_precomputed_ssp_mag_table,
        z_phot_table,
        nb_wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        nb_bin_centers_1d,
        nb_dmag,
        nb_mag_columns,
        nb_mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        nb_frac_cat,
        nb_z,
        nb_delta_z,
        lg_n_target_1d_nbs,
        lg_n_thresh,
    ) = nb_args

    # dynamic_slice_in_dim(arr, start, size, axis)
    nb_zval = lax.dynamic_slice_in_dim(nb_z, nb_idx, 1, axis=0)
    nb_zmin = nb_zval - (nb_delta_z / 2)
    nb_zmax = nb_zval + (nb_delta_z / 2)
    nb_z_weight = jnp.float64((lc_z_obs > nb_zmin) & (lc_z_obs <= nb_zmax))

    lg_n_model_1d_nb = n_mag_kern_nocolor(
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
        lax.dynamic_slice_in_dim(nb_precomputed_ssp_mag_table, nb_idx, 1, axis=1),
        z_phot_table,
        lax.dynamic_slice_in_dim(nb_wave_eff_table, nb_idx, 1, axis=1),
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        nb_bin_centers_1d,
        nb_dmag,
        lax.dynamic_slice_in_dim(nb_mag_columns, nb_idx, 1, axis=0),
        nb_mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        lax.dynamic_slice_in_dim(nb_frac_cat, nb_idx, 1, axis=0),
        nb_z_weight,
    )

    nb_in_lc = (nb_zval > lc_z_obs.min()) & (nb_zval <= lc_z_obs.max())

    nb_loss = jnp.where(
        nb_in_lc,
        _mse_w(
            lg_n_model_1d_nb[0][0],
            lg_n_target_1d_nbs[nb_idx][0],
            lg_n_target_1d_nbs[nb_idx][1],
            lg_n_thresh,
        ),
        0.0,
    )

    return nb_args, nb_loss


# Latin Hypercube bins based fitting + NBs separately in 1Ds
@jjit
def _loss_kern_w_nbs(
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
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params,
    fb,
    lg_n_target_1d_nbs,
    nb_z,
    nb_delta_z,
    nb_precomputed_ssp_mag_table,
    nb_wave_eff_table,
    nb_bin_centers_1d,
    nb_dmag,
    nb_mag_columns,
    nb_mag_thresh_column,
    nb_frac_cat,
    frac_cat=1.0,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_lc_z_obs=None,
    halpha_lc_t_obs=None,
    halpha_lc_mah_params=None,
    halpha_lc_nhalos=None,
    halpha_lc_vol_mpc3=None,
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
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
    )
    loss = _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1], lg_n_thresh)

    # Narrow band loss calculation
    nb_args = (
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
        nb_precomputed_ssp_mag_table,
        z_phot_table,
        nb_wave_eff_table,
        mzr_params,
        scatter_params,
        ssp_err_pop_params,
        nb_bin_centers_1d,
        nb_dmag,
        nb_mag_columns,
        nb_mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        nb_frac_cat,
        nb_z,
        nb_delta_z,
        lg_n_target_1d_nbs,
        lg_n_thresh,
    )
    nb_idx = jnp.arange(len(nb_z))
    _, nb_losses = lax.scan(compute_nb_loss, nb_args, nb_idx)
    loss += jnp.sum(nb_losses)

    if lg_halpha_LF_target is not None:
        halpha_lc_mah_params = DiffmahParams(*halpha_lc_mah_params)
        halpha_loss_args = (
            diffstarpop_params,
            ran_key,
            lg_halpha_LF_target,
            lg_halpha_Lbin_edges,
            lg_n_thresh,
            halpha_lc_z_obs,
            halpha_lc_t_obs,
            halpha_lc_mah_params,
            halpha_lc_nhalos,
            halpha_lc_vol_mpc3,
            t_table,
            ssp_data,
            ssp_halpha_luminosity,
            mzr_params,
            spspop_params,
            scatter_params,
            cosmo_params,
            fb,
        )
        loss += get_halpha_loss(*halpha_loss_args)

    return loss


_L_w_nbs = (
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
    None,
    None,
    0,
    0,
    0,
    None,
    None,
    0,
    0,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)
_loss_kern_w_nbs_multi_z = jjit(
    vmap(
        _loss_kern_w_nbs,
        in_axes=_L_w_nbs,
    )
)


@jjit
def _loss_w_nbs_total_multi_z(*args):
    return jnp.sum(_loss_kern_w_nbs_multi_z(*args))


loss_and_grad_w_nbs_multi_z = jjit(value_and_grad(_loss_w_nbs_total_multi_z))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_n_w_nbs_multi_z(
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
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    cosmo_params,
    fb,
    lg_n_target_1d_nbs,
    nb_z,
    nb_delta_z,
    nb_precomputed_ssp_mag_table,
    nb_wave_eff_table,
    nb_bin_centers_1d,
    nb_dmag,
    nb_mag_columns,
    nb_mag_thresh_column,
    nb_frac_cat,
    frac_cat=1.0,
    n_steps=2,
    step_size=0.1,
    ssp_halpha_luminosity=None,
    lg_halpha_LF_target=None,
    lg_halpha_Lbin_edges=None,
    halpha_lc_z_obs=None,
    halpha_lc_t_obs=None,
    halpha_lc_mah_params=None,
    halpha_lc_nhalos=None,
    halpha_lc_vol_mpc3=None,
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
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        lg_n_target_1d_nbs,
        nb_z,
        nb_delta_z,
        nb_precomputed_ssp_mag_table,
        nb_wave_eff_table,
        nb_bin_centers_1d,
        nb_dmag,
        nb_mag_columns,
        nb_mag_thresh_column,
        nb_frac_cat,
        frac_cat,
        ssp_halpha_luminosity,
        lg_halpha_LF_target,
        lg_halpha_Lbin_edges,
        halpha_lc_z_obs,
        halpha_lc_t_obs,
        halpha_lc_mah_params,
        halpha_lc_nhalos,
        halpha_lc_vol_mpc3,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_w_nbs_multi_z(u_theta, *other)
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
