import jax.numpy as jnp
from diffsky.experimental.scatter import DEFAULT_SCATTER_U_PARAMS
from dsps.metallicity.umzr import DEFAULT_MZR_U_PARAMS
from jax import jit as jjit

from ..kernels.spec_kern import n_spec_kern
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import mse_w


@jjit
def get_emline_loss(
    ran_key,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
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

    emline_loss = mse_w(
        lg_emline_LF_model,
        lg_emline_LF_target[0],
        lg_emline_LF_target[1],
    )

    return emline_loss


def _loss_emline_kern(
    u_theta,
    ran_key,
    lg_emline_LF_target,
    lg_emline_Lbin_edges,
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
