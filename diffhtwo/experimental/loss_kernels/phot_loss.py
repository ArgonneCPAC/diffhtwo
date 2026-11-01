from jax import jit as jjit
from jax import lax

from ..kernels.N_phot import add_N_model_to_spaces
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import poisson_loss


@jjit
def get_phot_loss_2d_multiz(ran_key, param_collection, data, mag_thresh):
    phot_loss_2d = 0.0
    n_z_bins = len(data)
    for z in range(n_z_bins):
        z_data = data[z]
        z_data_model = add_N_model_to_spaces(
            ran_key,
            param_collection,
            z_data,
            mag_thresh,
        )
        space_names = [
            f for f in z_data_model._fields if f not in {"z_min", "z_max", "lc_data"}
        ]
        for space_name in space_names:
            spaces = getattr(z_data_model, space_name)
            if not isinstance(spaces, list):
                spaces = [spaces]
            for space in spaces:
                phot_loss_2d += lax.cond(
                    space.fit,
                    lambda sp=space: poisson_loss(
                        sp.N_model / sp.vol_mpc3_model,
                        sp.N_data / sp.vol_mpc3_data,
                    ),
                    lambda: 0.0,
                )
    return phot_loss_2d


@jjit
def _loss_phot_kern_2d_multiz(u_theta, ran_key, fitting_data):
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_2d = 0.0

    # get loss by going through all the fitted spaces: mag, color-color, and color-magnitude
    phot_loss_2d += get_phot_loss_2d_multiz(
        ran_key,
        param_collection,
        fitting_data.spaces,
        fitting_data.filter_info.mag_thresh,
    )

    return phot_loss_2d
