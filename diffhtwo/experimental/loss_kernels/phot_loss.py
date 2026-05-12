from jax import jit as jjit

from ..kernels.N_phot import N_colors_mags_lh
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
