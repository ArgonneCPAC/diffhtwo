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

from ..n_specphot import n_colors_mags_lh, n_spec_kern
from ..param_utils import get_param_collection_from_u_theta

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_zero_ssperrpop_theta, u_zero_ssperrpop_unravel = ravel_pytree(ZERO_SSPERR_U_PARAMS)


@jjit
def _mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh):
    mask = lg_n_target > lg_n_thresh
    nbins = jnp.maximum(jnp.sum(mask), 1)

    resid = (lg_n_pred - lg_n_target) ** 2
    chi2 = resid / lg_n_target_err
    chi2 = jnp.where(mask, chi2, 0.0)

    return jnp.sum(chi2) / nbins


@partial(jjit, static_argnames=["redshift_as_last_dimension_in_lh"])
def get_phot_loss(
    ran_key,
    lg_n_target,
    lg_n_thresh,
    param_collection,
    lc_data,
    line_wave_aa,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    redshift_as_last_dimension_in_lh=False,
):
    line_wave_table = jnp.array([line_wave_aa])
    n_colors_mags_lh_args = (
        ran_key,
        param_collection,
        lc_data,
        line_wave_table,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
        redshift_as_last_dimension_in_lh,
    )
    lg_n_model, _ = n_colors_mags_lh(*n_colors_mags_lh_args)
    phot_loss = _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1], lg_n_thresh)

    return phot_loss


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
    lg_n_target,
    lg_n_thresh,
    lc_data,
    line_wave_aa,
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
        lg_n_target,
        lg_n_thresh,
        param_collection,
        lc_data,
        line_wave_aa,
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


_L_pk = (None, None, 0, None, 0, None, None, None, None, 0, 0, None)
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
    lg_n_thresh,
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
                lg_n_thresh,
                emline_lc_data[line][z],
                emline_wave_table[line],
            )
            emline_loss_multi_line_multi_z += _loss_emline_kern(*emline_loss_args_z)

    return emline_loss_multi_line_multi_z


@jjit
def _loss_phot_and_emline_multi_z(
    u_theta,
    ran_key,
    lg_n_target,
    lg_n_thresh,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    emline_lc_data,
    emline_wave_table,
    redshift_as_last_dimension_in_lh=False,
):
    emline_wave_aa = emline_wave_table[0]
    phot_multi_z_loss_args = (
        u_theta,
        ran_key,
        lg_n_target,
        lg_n_thresh,
        lc_data,
        emline_wave_aa,  # dummy arg
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
        redshift_as_last_dimension_in_lh,
    )
    phot_loss_multi_z = _loss_phot_kern_multi_z(*phot_multi_z_loss_args)

    emline_multi_line_multi_z_loss_args = (
        u_theta,
        ran_key,
        lg_n_thresh,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        emline_lc_data,
        emline_wave_table,
    )
    emline_loss_multi_line_multi_z = _loss_emline_kern_multi_line_multi_z(
        *emline_multi_line_multi_z_loss_args
    )

    phot_and_emline_loss_multi_z = (
        jnp.sum(phot_loss_multi_z) + emline_loss_multi_line_multi_z
    )
    return phot_and_emline_loss_multi_z


loss_and_grad_phot_and_emline_multi_z = jjit(
    value_and_grad(_loss_phot_and_emline_multi_z)
)


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_phot_and_emline_multi_z(
    u_theta_init,
    trainable,
    ran_key,
    lg_n_target,
    lg_n_thresh,
    lc_data,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lh_centroids,
    d_centroids,
    frac_cat,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    emline_lc_data,
    emline_wave_table,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        lg_n_target,
        lg_n_thresh,
        lc_data,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        emline_lc_data,
        emline_wave_table,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = loss_and_grad_phot_and_emline_multi_z(u_theta, *other)
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
def _loss_sdss_feniks_hizels(
    u_theta,
    ran_key,
    lg_n_thresh,
    sdss_lg_n_target,
    sdss_lc_data,
    sdss_mag_columns,
    sdss_mag_thresh_column,
    sdss_mag_thresh,
    sdss_lh_centroids,
    sdss_d_centroids,
    sdss_frac_cat,
    feniks_lg_n_target,
    feniks_lc_data,
    feniks_mag_columns,
    feniks_mag_thresh_column,
    feniks_mag_thresh,
    feniks_lh_centroids,
    feniks_d_centroids,
    feniks_frac_cat,
    hizels_lg_LF_target,
    hizels_lg_Lbin_edges,
    hizels_lc_data,
    hizels_wave_table,
):
    emline_wave_aa = hizels_wave_table[0]

    # sdss
    sdss_phot_loss_args = (
        u_theta,
        ran_key,
        sdss_lg_n_target,
        lg_n_thresh,
        sdss_lc_data,
        emline_wave_aa,  # dummy arg
        sdss_mag_columns,
        sdss_mag_thresh_column,
        sdss_mag_thresh,
        sdss_lh_centroids,
        sdss_d_centroids,
        sdss_frac_cat,
    )
    sdss_phot_loss = _loss_phot_kern(
        *sdss_phot_loss_args, redshift_as_last_dimension_in_lh=True
    )

    # feniks
    feniks_phot_multi_z_loss_args = (
        u_theta,
        ran_key,
        feniks_lg_n_target,
        lg_n_thresh,
        feniks_lc_data,
        emline_wave_aa,  # dummy arg
        feniks_mag_columns,
        feniks_mag_thresh_column,
        feniks_mag_thresh,
        feniks_lh_centroids,
        feniks_d_centroids,
        feniks_frac_cat,
    )
    feniks_phot_loss_multi_z = _loss_phot_kern_multi_z(*feniks_phot_multi_z_loss_args)

    # hizels
    hizels_emline_multi_line_multi_z_loss_args = (
        u_theta,
        ran_key,
        lg_n_thresh,
        hizels_lg_LF_target,
        hizels_lg_Lbin_edges,
        hizels_lc_data,
        hizels_wave_table,
    )
    hizels_emline_loss = _loss_emline_kern_multi_line_multi_z(
        *hizels_emline_multi_line_multi_z_loss_args
    )

    sdss_feniks_hizels_loss = (
        sdss_phot_loss + jnp.sum(feniks_phot_loss_multi_z) + hizels_emline_loss
    )
    return sdss_feniks_hizels_loss


loss_and_grad_sdss_feniks_hizels = jjit(value_and_grad(_loss_sdss_feniks_hizels))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    lg_n_thresh,
    sdss_lg_n_target,
    sdss_lc_data,
    sdss_mag_columns,
    sdss_mag_thresh_column,
    sdss_mag_thresh,
    sdss_lh_centroids,
    sdss_d_centroids,
    sdss_frac_cat,
    feniks_lg_n_target,
    feniks_lc_data,
    feniks_mag_columns,
    feniks_mag_thresh_column,
    feniks_mag_thresh,
    feniks_lh_centroids,
    feniks_d_centroids,
    feniks_frac_cat,
    hizels_lg_LF_target,
    hizels_lg_Lbin_edges,
    hizels_lc_data,
    hizels_wave_table,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        lg_n_thresh,
        sdss_lg_n_target,
        sdss_lc_data,
        sdss_mag_columns,
        sdss_mag_thresh_column,
        sdss_mag_thresh,
        sdss_lh_centroids,
        sdss_d_centroids,
        sdss_frac_cat,
        feniks_lg_n_target,
        feniks_lc_data,
        feniks_mag_columns,
        feniks_mag_thresh_column,
        feniks_mag_thresh,
        feniks_lh_centroids,
        feniks_d_centroids,
        feniks_frac_cat,
        hizels_lg_LF_target,
        hizels_lg_Lbin_edges,
        hizels_lc_data,
        hizels_wave_table,
    )

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
