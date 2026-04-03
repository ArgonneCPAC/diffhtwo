# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
from functools import partial

import jax.numpy as jnp
from diffhalos.lightcone_generators.mc_lightcone_halos import weighted_lc_halos
from diffmah.defaults import DiffmahParams
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental.lightcone_generators import weighted_lc_halos_photdata
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

from .. import emline_luminosity
from ..n_mag import N_0, N_FLOOR, n_mag_kern, n_mag_kern_1d, n_mag_kern_nocolor

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)
u_zero_ssperrpop_theta, u_zero_ssperrpop_unravel = ravel_pytree(ZERO_SSPERR_U_PARAMS)


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


@jjit
def get_phot_loss(
    diffstarpop_params,
    spspop_params,
    ssperrpop_params,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lc_z_min,
    lc_z_max,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    cosmo_params,
    fb,
    frac_cat,
    num_halos=1000,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    sky_area_degsq=0.1,
):
    # generate lightcone and photometry data
    lc_halopop = weighted_lc_halos(
        ran_key, num_halos, lc_z_min, lc_z_max, lgmp_min, lgmp_max, sky_area_degsq
    )

    lg_n_model, _ = n_mag_kern(
        diffstarpop_params,
        spspop_params,
        ran_key,
        lc_halopop.z_obs,
        lc_halopop.t_obs,
        lc_halopop.mah_params,
        lc_halopop.logmp0,
        lc_halopop.nhalos,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        scatter_params,
        ssperrpop_params,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        cosmo_params,
        fb,
        frac_cat,
    )
    phot_loss = _mse_w(lg_n_model, lg_n_target[0], lg_n_target[1], lg_n_thresh)

    return phot_loss


@jjit
def get_emline_loss(
    ran_key,
    emline_wave_aa,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    lg_n_thresh,
    lc_z_min,
    lc_z_max,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    diffstarpop_params,
    spspop_params,
    mzr_params,
    scatter_params,
    cosmo_params,
    fb,
    num_halos=1000,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    sky_area_degsq=0.1,
):
    lc_halopop = weighted_lc_halos(
        ran_key, num_halos, lc_z_min, lc_z_max, lgmp_min, lgmp_max, sky_area_degsq
    )
    L_emline_cgs, _ = emline_luminosity.compute_emline_luminosity(
        ran_key,
        lc_halopop.z_obs,
        lc_halopop.t_obs,
        lc_halopop.mah_params,
        diffstarpop_params,
        spspop_params,
        mzr_params,
        scatter_params,
        t_table,
        ssp_data,
        emline_wave_aa,
        cosmo_params,
        fb,
    )

    sig = jnp.diff(lg_emline_Lbin_edges) / 2
    sig = sig.reshape(sig.size, 1)
    _, emline_N = emline_luminosity.get_emline_luminosity_func(
        L_emline_cgs, lc_halopop.nhalos, sig=sig, lgL_bin_edges=lg_emline_Lbin_edges
    )
    # take care of bins with low/zero number counts in a similar way to n_mag.get_n_data_err(), using same N_floor and N_0:
    emline_N = jnp.where(emline_N > N_FLOOR, emline_N, N_0)

    lg_emline_LF_model = jnp.log10(emline_N / lc_vol_mpc3)

    return _mse_w(
        lg_emline_LF_model,
        lg_emline_LF_target[0],
        lg_emline_LF_target[1],
        lg_n_thresh,
    )


@jjit
def _loss_phot_kern(
    u_theta,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lc_z_min,
    lc_z_max,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    cosmo_params,
    fb,
    frac_cat,
):
    # get bounded params
    u_diffstarpop_theta, u_spspop_theta, u_ssperrpop_theta = u_theta

    u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
    diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

    u_spspop_params = u_spspop_unravel(u_spspop_theta)
    spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)

    u_ssperrpop_params = u_zero_ssperrpop_unravel(u_ssperrpop_theta)
    ssperrpop_params = get_bounded_ssperr_params(u_ssperrpop_params)

    phot_loss_args = (
        diffstarpop_params,
        spspop_params,
        ssperrpop_params,
        lg_n_target,
        lg_n_thresh,
        ran_key,
        mzr_params,
        scatter_params,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min,
        lc_z_max,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        cosmo_params,
        fb,
        frac_cat,
    )
    phot_loss = get_phot_loss(*phot_loss_args)

    return phot_loss


_L_pk = (
    None,
    0,
    None,
    None,
    None,
    None,
    0,
    0,
    None,
    None,
    None,
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
    emline_wave_aa,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    lg_n_thresh,
    emline_lc_z_min,
    emline_lc_z_max,
    emline_lc_vol_mpc3,
    t_table,
    ssp_data,
    mzr_params,
    scatter_params,
    cosmo_params,
    fb,
):
    # get bounded params
    u_diffstarpop_theta, u_spspop_theta, u_ssperrpop_theta = u_theta

    u_diffstarpop_params = u_diffstarpop_unravel(u_diffstarpop_theta)
    diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

    u_spspop_params = u_spspop_unravel(u_spspop_theta)
    spspop_params = get_bounded_spspop_params_tw_dust(u_spspop_params)

    u_ssperrpop_params = u_zero_ssperrpop_unravel(u_ssperrpop_theta)
    ssperrpop_params = get_bounded_ssperr_params(u_ssperrpop_params)

    emline_loss_args = (
        ran_key,
        emline_wave_aa,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        lg_n_thresh,
        emline_lc_z_min,
        emline_lc_z_max,
        emline_lc_vol_mpc3,
        t_table,
        ssp_data,
        diffstarpop_params,
        spspop_params,
        mzr_params,
        scatter_params,
        cosmo_params,
        fb,
    )
    emline_loss = get_emline_loss(*emline_loss_args)
    return emline_loss


@jjit
def _loss_phot_and_emline_multi_z(
    u_theta,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lc_z_min,
    lc_z_max,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    cosmo_params,
    fb,
    frac_cat,
    emline_wave_aa,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    emline_lc_z_min,
    emline_lc_z_max,
    emline_lc_vol_mpc3,
):
    phot_multi_z_loss_args = (
        u_theta,
        lg_n_target,
        lg_n_thresh,
        ran_key,
        mzr_params,
        scatter_params,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min,
        lc_z_max,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        cosmo_params,
        fb,
        frac_cat,
    )
    phot_loss_multi_z = _loss_phot_kern_multi_z(*phot_multi_z_loss_args)

    emline_loss_multi_z = 0.0

    n_line = len(emline_wave_aa)
    for line in range(0, n_line):
        n_z = len(lg_emline_LF_target[line])
        for z in range(0, n_z):
            emline_loss_args_z = (
                u_theta,
                ran_key,
                emline_wave_aa[line],
                lg_emline_LF_target[line][z],
                lg_emline_Lbin_edges[line][z],
                lg_n_thresh,
                emline_lc_z_min[line][z],
                emline_lc_z_max[line][z],
                emline_lc_vol_mpc3[line][z],
                t_table,
                ssp_data,
                mzr_params,
                scatter_params,
                cosmo_params,
                fb,
            )
            emline_loss_multi_z += _loss_emline_kern(*emline_loss_args_z)

    phot_and_emline_loss_multi_z = jnp.sum(phot_loss_multi_z) + emline_loss_multi_z
    return phot_and_emline_loss_multi_z


loss_and_grad_phot_and_emline_multi_z = jjit(
    value_and_grad(_loss_phot_and_emline_multi_z)
)


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_phot_and_emline_multi_z(
    u_theta_init,
    trainable,
    lg_n_target,
    lg_n_thresh,
    ran_key,
    mzr_params,
    scatter_params,
    lh_centroids,
    dmag_centroids,
    mag_columns,
    mag_thresh_column,
    mag_thresh,
    lc_z_min,
    lc_z_max,
    lc_vol_mpc3,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    cosmo_params,
    fb,
    frac_cat,
    emline_wave_aa,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
    emline_lc_z_min,
    emline_lc_z_max,
    emline_lc_vol_mpc3,
    n_steps=2,
    step_size=0.1,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        lg_n_target,
        lg_n_thresh,
        ran_key,
        mzr_params,
        scatter_params,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min,
        lc_z_max,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        cosmo_params,
        fb,
        frac_cat,
        emline_wave_aa,
        lg_emline_LF_target,
        lg_emline_Lbin_edges,
        emline_lc_z_min,
        emline_lc_z_max,
        emline_lc_vol_mpc3,
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
