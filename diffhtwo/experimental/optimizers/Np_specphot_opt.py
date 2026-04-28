# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from functools import partial

import jax.numpy as jnp
from diffsky.experimental.scatter import DEFAULT_SCATTER_U_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_U_PARAMS
from diffsky.ssp_err_model.defaults import ZERO_SSPERR_U_PARAMS
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from dsps.metallicity.umzr import DEFAULT_MZR_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad, vmap
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from .. import diffndhist as diffndhist2
from ..n_specphot import n_spec_kern, phot_kern
from ..Np_specphot import N_colors_mags_lh
from ..param_utils import get_param_collection_from_u_theta

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_zero_ssperrpop_theta, u_zero_ssperrpop_unravel = ravel_pytree(ZERO_SSPERR_U_PARAMS)


@jjit
def poisson_loss(N_pred, N_target, eps=1e-12):
    N_pred = jnp.clip(N_pred, eps, None)
    return jnp.sum(N_pred - N_target * jnp.log(N_pred))


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def get_phot_loss(
    ran_key,
    N_target,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    redshift_as_last_dimension_in_lh=False,
):
    N_colors_mags_lh_args = (
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
        redshift_as_last_dimension_in_lh,
    )
    N_model = N_colors_mags_lh(*N_colors_mags_lh_args)
    phot_loss = poisson_loss(N_model, N_target)
    return phot_loss


@jjit
def get_mag_func_loss(
    ran_key,
    param_collection,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    frac_cat,
    mag_bin_edges,
    N_target,
):
    mags, weights = phot_kern(
        ran_key,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        frac_cat,
    )

    mags = mags[:, -1]
    mags = mags.reshape(mags.size, 1)

    bw = jnp.diff(mag_bin_edges).mean()

    mag_lo = mag_bin_edges[:-1]
    mag_lo = mag_lo.reshape(mag_lo.size, 1)

    mag_hi = mag_bin_edges[1:]
    mag_hi = mag_lo.reshape(mag_hi.size, 1)

    sig = jnp.zeros(mag_lo.shape) + (bw / 2)
    mag_bin_edges = mag_bin_edges.reshape(mag_bin_edges.size, 1)

    N_model = diffndhist2.tw_ndhist_weighted(
        mags,
        sig,
        weights,
        mag_lo,
        mag_hi,
    )

    mag_func_loss = poisson_loss(N_model, N_target)

    return mag_func_loss


@jjit
def get_emline_loss(
    ran_key,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    lg_n_thresh,
    param_collection,
    lc_data,
    line_wave_aa,
):
    line_wave_table = jnp.array([line_wave_aa])
    lg_emline_LF_model = n_spec_kern(
        ran_key,
        param_collection,
        lc_data,
        line_wave_table,
        lg_emline_Lbin_edges,
    )

    emline_loss = _mse_w(
        lg_emline_LF_model,
        lg_emline_LF_target[0],
        lg_emline_LF_target[1],
        lg_n_thresh,
    )

    return emline_loss


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def _loss_phot_kern(
    u_theta,
    ran_key,
    N_target,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    redshift_as_last_dimension_in_lh=False,
):
    # get bounded param collection
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_args = (
        ran_key,
        N_target,
        param_collection,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
        redshift_as_last_dimension_in_lh,
    )
    phot_loss = get_phot_loss(*phot_loss_args)

    return phot_loss


_L_pk = (
    None,
    None,
    0,
    0,
    None,
    None,
    None,
    0,
    0,
    None,
)
_loss_phot_kern_multi_z = jjit(
    vmap(
        _loss_phot_kern,
        in_axes=_L_pk,
    )
)


def _loss_emline_kern(
    u_theta,
    ran_key,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    lg_n_thresh,
    lc_data,
    line_wave_aa,
    u_mzr_params=DEFAULT_MZR_U_PARAMS,
    u_scatter_params=DEFAULT_SCATTER_U_PARAMS,
):
    param_collection = get_param_collection_from_u_theta(u_theta)
    emline_loss_args = (
        ran_key,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        lg_n_thresh,
        param_collection,
        lc_data,
        line_wave_aa,
    )
    emline_loss = get_emline_loss(*emline_loss_args)
    return emline_loss


@jjit
def _loss_emline_kern_multi_line_multi_z(
    u_theta,
    ran_key,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    emline_lc_data,
    emline_wave_table,
):
    emline_loss_multi_line_multi_z = 0.0

    n_line = len(emline_wave_table)
    for line in range(0, n_line):
        n_z = len(lg_emline_LF_target[line])
        for z in range(0, n_z):
            emline_loss_args_z = (
                u_theta,
                ran_key,
                lg_emline_LF_target[line][z],
                lg_emline_Lbin_edges[line][z],
                emline_lc_data[line][z],
                emline_wave_table[line],
            )
            emline_loss_multi_line_multi_z += _loss_emline_kern(*emline_loss_args_z)

    return emline_loss_multi_line_multi_z


@jjit
def _loss_sdss_feniks_hizels(
    u_theta,
    ran_key,
    sdss,
    feniks,
    hizels,
    line_wave_table,
    fit_hizels=False,
    fit_sdss_mag_thresh_band=False,
    fit_feniks_mag_thresh_band=False,
):
    # sdss
    sdss_phot_loss_args = (
        u_theta,
        ran_key,
        sdss.lg_n_data_err_lh,
        sdss.lc_data,
        sdss.mag_columns,
        sdss.mag_thresh_column,
        sdss.mag_thresh,
        sdss.lh_centroids,
        sdss.d_centroids,
        sdss.frac_cat,
    )
    sdss_phot_loss = _loss_phot_kern(
        *sdss_phot_loss_args, redshift_as_last_dimension_in_lh=True
    )
    if fit_sdss_mag_thresh_band:
        param_collection = get_param_collection_from_u_theta(u_theta)
        mag_func_loss_args = (
            ran_key,
            param_collection,
            sdss.lc_data,
            sdss.mag_columns,
            sdss.mag_thresh_column,
            sdss.mag_thresh,
            sdss.frac_cat,
            sdss.norm_band_bin_edges,
            sdss.norm_band_lg_n,
            sdss.norm_band_lg_n_avg_err,
        )
        sdss_mag_func_loss = get_mag_func_loss(*mag_func_loss_args)
        sdss_phot_loss = sdss_phot_loss + sdss_mag_func_loss

    # feniks
    feniks_phot_loss_args = (
        u_theta,
        ran_key,
        feniks.lg_n_data_err_lh,
        feniks.lc_data,
        feniks.mag_columns,
        feniks.mag_thresh_column,
        feniks.mag_thresh,
        feniks.lh_centroids,
        feniks.d_centroids,
        feniks.frac_cat,
    )
    feniks_phot_loss = _loss_phot_kern(
        *feniks_phot_loss_args, redshift_as_last_dimension_in_lh=True
    )
    if fit_feniks_mag_thresh_band:
        param_collection = get_param_collection_from_u_theta(u_theta)
        mag_func_loss_args = (
            ran_key,
            param_collection,
            feniks.lc_data,
            feniks.mag_columns,
            feniks.mag_thresh_column,
            feniks.mag_thresh,
            feniks.frac_cat,
            feniks.norm_band_bin_edges,
            feniks.norm_band_lg_n,
            feniks.norm_band_lg_n_avg_err,
        )
        feniks_mag_func_loss = get_mag_func_loss(*mag_func_loss_args)
        feniks_phot_loss = feniks_phot_loss + feniks_mag_func_loss

    # hizels
    if fit_hizels:
        hizels_emline_multi_line_multi_z_loss_args = (
            u_theta,
            ran_key,
            hizels.lg_LF,
            hizels.lg_Lbin_edges,
            hizels.lc_data,
            line_wave_table,
        )
        hizels_emline_loss = _loss_emline_kern_multi_line_multi_z(
            *hizels_emline_multi_line_multi_z_loss_args
        )
    else:
        hizels_emline_loss = 0.0

    sdss_feniks_hizels_loss = sdss_phot_loss + feniks_phot_loss + hizels_emline_loss
    return sdss_feniks_hizels_loss


loss_and_grad_sdss_feniks_hizels = jjit(value_and_grad(_loss_sdss_feniks_hizels))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    sdss,
    feniks,
    hizels,
    line_wave_table,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (ran_key, sdss, feniks, hizels, line_wave_table)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_sdss_feniks_hizels(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit


@jjit
def _loss_sdss_or_feniks(u_theta, ran_key, dataset, fit_mag_thresh_band=False):
    # dataset
    loss_phot_kern_args = (
        u_theta,
        ran_key,
        dataset.N_data,
        dataset.lc_data,
        dataset.mag_columns,
        dataset.mag_thresh_column,
        dataset.mag_thresh,
        dataset.lh_centroids,
        dataset.d_centroids,
        dataset.frac_cat,
    )
    phot_loss = _loss_phot_kern(
        *loss_phot_kern_args, redshift_as_last_dimension_in_lh=True
    )
    if fit_mag_thresh_band:
        param_collection = get_param_collection_from_u_theta(u_theta)
        mag_func_loss_args = (
            ran_key,
            param_collection,
            dataset.lc_data,
            dataset.mag_columns,
            dataset.mag_thresh_column,
            dataset.mag_thresh,
            dataset.frac_cat,
            dataset.norm_band_bin_edges,
            dataset.norm_band_lg_n,
            dataset.norm_band_lg_n_avg_err,
        )
        mag_func_loss = get_mag_func_loss(*mag_func_loss_args)
        return phot_loss + mag_func_loss
    else:
        return phot_loss


loss_and_grad_sdss_or_feniks = jjit(value_and_grad(_loss_sdss_or_feniks))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_or_feniks(
    u_theta_init,
    trainable,
    ran_key,
    dataset,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        dataset,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_sdss_or_feniks(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit
