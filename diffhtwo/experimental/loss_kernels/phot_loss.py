import jax.numpy as jnp
from jax import jit as jjit

from .. import diffndhist as diffndhist2
from ..kernels.Np_phot import N_colors_mags_lh
from ..kernels.phot_kern import mag_kern
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import poisson_loss


@jjit
def get_phot_loss(
    ran_key,
    meta_data,
    fitting_data,
    param_collection,
):
    N_colors_mags_lh_args = (
        ran_key,
        meta_data,
        fitting_data,
        param_collection,
    )
    N_model = N_colors_mags_lh(*N_colors_mags_lh_args)
    N_model = N_model * (
        meta_data.data_sky_area_degsq / fitting_data.lc_data.sky_area_degsq
    )
    phot_loss = poisson_loss(N_model, fitting_data.N_data)
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
    mags, weights = mag_kern(
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
def _loss_phot_kern(
    u_theta,
    ran_key,
    meta_data,
    fitting_data,
):
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_args = (
        ran_key,
        meta_data,
        fitting_data,
        param_collection,
    )
    phot_loss = get_phot_loss(*phot_loss_args)

    return phot_loss
