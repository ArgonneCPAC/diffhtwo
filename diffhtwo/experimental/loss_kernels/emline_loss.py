import jax.numpy as jnp
from diffsky.experimental.scatter import DEFAULT_SCATTER_U_PARAMS
from dsps.metallicity.umzr import DEFAULT_MZR_U_PARAMS
from jax import jit as jjit

from ..kernels.N_spec import N_linelum
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import poisson_loss


@jjit
def get_emline_loss(
    ran_key,
    line_wave_aa,
    lg_Lbin_edges,
    N_data,
    vol_Mpc3_data,
    lc_data,
    param_collection,
):
    line_wave_table = jnp.array([line_wave_aa])

    N_model = N_linelum(
        ran_key,
        line_wave_table,
        lg_Lbin_edges,
        lc_data,
        param_collection,
    )

    # N_model = N_model * (vol_Mpc3_data / lc_data.lc_tot_vol_mpc3)
    n_model = N_model / lc_data.lc_tot_vol_mpc3
    n_data = N_data / vol_Mpc3_data

    emline_loss = poisson_loss(n_model, n_data)

    return emline_loss


def _loss_emline_kern(
    u_theta,
    ran_key,
    line_wave_aa,
    lg_Lbin_edges,
    N_data,
    vol_Mpc3_data,
    lc_data,
    u_mzr_params=DEFAULT_MZR_U_PARAMS,
    u_scatter_params=DEFAULT_SCATTER_U_PARAMS,
):
    param_collection = get_param_collection_from_u_theta(u_theta)
    emline_loss_args = (
        ran_key,
        line_wave_aa,
        lg_Lbin_edges,
        N_data,
        vol_Mpc3_data,
        lc_data,
        param_collection,
    )
    emline_loss = get_emline_loss(*emline_loss_args)
    return emline_loss


@jjit
def _loss_emline_kern_multi_line_multi_z(
    u_theta,
    ran_key,
    fitting_data_multi_line_multi_z,
):
    emline_loss_multi_line_multi_z = 0.0

    n_line = len(fitting_data_multi_line_multi_z.lg_Lbin_edges)
    for line in range(0, n_line):
        n_z = len(fitting_data_multi_line_multi_z.lg_Lbin_edges[line])
        for z in range(0, n_z):
            emline_loss_args_z = (
                u_theta,
                ran_key,
                fitting_data_multi_line_multi_z.line_wave_aa[line],
                fitting_data_multi_line_multi_z.lg_Lbin_edges[line][z],
                fitting_data_multi_line_multi_z.N_data[line][z],
                fitting_data_multi_line_multi_z.vol_Mpc3_data[line][z],
                fitting_data_multi_line_multi_z.lc_data[line][z],
            )
            emline_loss_multi_line_multi_z += _loss_emline_kern(*emline_loss_args_z)

    return emline_loss_multi_line_multi_z
