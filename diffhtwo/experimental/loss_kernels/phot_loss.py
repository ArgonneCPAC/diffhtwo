from jax import jit as jjit
from jax import lax

from ..kernels.N_phot import N_colors_mags, N_colors_mags_lh
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import poisson_loss


@jjit
def get_phot_loss_2d_multiz(
    ran_key,
    param_collection,
    data,
    mag_thresh,
    frac_cat,
):
    phot_loss_2d = 0.0
    for z in range(0, len(data)):
        z_data = data[z]
        z_data_model = N_colors_mags(
            ran_key,
            param_collection,
            z_data,
            mag_thresh,
            frac_cat,
        )
        # sky_rescale = data_sky_area_degsq / z_data_model.lc_data.sky_area_degsq
        fields = z_data_model._fields[3:]
        for f in range(0, len(fields)):
            space = getattr(z_data_model, fields[f])
            if isinstance(space, list):
                for s in range(0, len(space)):
                    space_n = space[s]
                    phot_loss_2d += lax.cond(
                        space_n.fit,
                        lambda sp=space_n: poisson_loss(
                            sp.N_model / z_data_model.lc_data.lc_tot_vol_mpc3,
                            sp.N_data / z_data_model.data_vol_mpc3,
                        ),
                        lambda: 0.0,
                    )
            else:
                phot_loss_2d += lax.cond(
                    space.fit,
                    lambda sp=space: poisson_loss(
                        sp.N_model / z_data_model.lc_data.lc_tot_vol_mpc3,
                        sp.N_data / z_data_model.data_vol_mpc3,
                    ),
                    lambda: 0.0,
                )
    return phot_loss_2d


@jjit
def _loss_phot_kern_2d_multiz(
    u_theta,
    ran_key,
    fitting_data,
):
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_2d = 0.0

    # get color loss
    phot_loss_2d += get_phot_loss_2d_multiz(
        ran_key,
        param_collection,
        fitting_data.colors,
        fitting_data.filter_info.mag_thresh,
        fitting_data.frac_cat,
    )
    # get app mag func loss
    phot_loss_2d += get_phot_loss_2d_multiz(
        ran_key,
        param_collection,
        fitting_data.app_mag_funcs,
        fitting_data.filter_info.mag_thresh,
        fitting_data.frac_cat,
    )

    return phot_loss_2d


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
