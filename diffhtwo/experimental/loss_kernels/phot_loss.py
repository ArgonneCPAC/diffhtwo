from jax import jit as jjit

from ..kernels.N_phot import N_colors_mags, N_colors_mags_lh, N_mags_1d
from ..param_utils import get_param_collection_from_u_theta
from .loss_functions import poisson_loss


@jjit
def get_phot_loss_2d(
    ran_key,
    param_collection,
    z_data,
    mag_thresh,
    frac_cat,
    data_sky_area_degsq,
):
    z_data = N_colors_mags(
        ran_key,
        param_collection,
        z_data,
        mag_thresh,
        frac_cat,
    )
    phot_loss_2d = 0.0
    fields = z_data._fields[3:]
    for f in range(0, len(fields)):
        data = getattr(z_data, fields[f])

        if isinstance(data, list):
            for d in range(0, len(data)):
                data_n = data[d]

                N_model = data_n.N_model
                N_data = data_n.N_data

                N_model = N_model * (
                    data_sky_area_degsq / z_data.lc_data.sky_area_degsq
                )
                phot_loss_2d += poisson_loss(N_model, N_data)

        else:
            N_model = data.N_model
            N_data = data.N_data

            N_model = N_model * (data_sky_area_degsq / z_data.lc_data.sky_area_degsq)
            phot_loss_2d += poisson_loss(N_model, N_data)

    return phot_loss_2d


@jjit
def _loss_phot_kern_2d(
    u_theta,
    ran_key,
    z_data,
    mag_thresh,
    frac_cat,
    data_sky_area_degsq,
):
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_2d = get_phot_loss_2d(
        ran_key,
        param_collection,
        z_data,
        mag_thresh,
        frac_cat,
        data_sky_area_degsq,
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

    phot_loss_2d += get_phot_loss_2d(
        ran_key,
        param_collection,
        fitting_data.z1,
        fitting_data.filter_info.mag_thresh,
        fitting_data.frac_cat,
        fitting_data.data_sky_area_degsq,
    )
    phot_loss_2d += get_phot_loss_2d(
        ran_key,
        param_collection,
        fitting_data.z2,
        fitting_data.filter_info.mag_thresh,
        fitting_data.frac_cat,
        fitting_data.data_sky_area_degsq,
    )
    phot_loss_2d += get_phot_loss_2d(
        ran_key,
        param_collection,
        fitting_data.z3,
        fitting_data.filter_info.mag_thresh,
        fitting_data.frac_cat,
        fitting_data.data_sky_area_degsq,
    )

    return phot_loss_2d


@jjit
def get_phot_loss_1d(
    ran_key,
    param_collection,
    magbin_bands,
    N_bands_data,
    lc_data,
    mag_thresh,
    frac_cat,
    data_sky_area_degsq,
):
    N_bands_model = N_mags_1d(
        ran_key, param_collection, magbin_bands, lc_data, mag_thresh, frac_cat
    )

    n_bands = len(N_bands_data)
    phot_loss_1d = 0.0
    for band in range(0, n_bands):
        N_model = N_bands_model[band] * (data_sky_area_degsq / lc_data.sky_area_degsq)
        phot_loss_1d += poisson_loss(N_model, N_bands_data[band])

    return phot_loss_1d


@jjit
def _loss_phot_kern_1d(
    u_theta,
    ran_key,
    magbin_bands,
    N_bands_data,
    lc_data,
    mag_thresh,
    frac_cat,
    data_sky_area_degsq,
):
    param_collection = get_param_collection_from_u_theta(u_theta)

    phot_loss_1d_args = (
        ran_key,
        param_collection,
        magbin_bands,
        N_bands_data,
        lc_data,
        mag_thresh,
        frac_cat,
        data_sky_area_degsq,
    )
    phot_loss_1d = get_phot_loss_1d(*phot_loss_1d_args)

    return phot_loss_1d


@jjit
def _loss_phot_kern_multiband_multiz(
    u_theta,
    ran_key,
    fitting_data,
):
    zbins = fitting_data.zbins
    n_z_bins = len(zbins)
    phot_loss_multiband_multiz = 0.0
    for zbin in range(n_z_bins):
        phot_loss_multiband_multiz += _loss_phot_kern_1d(
            u_theta,
            ran_key,
            fitting_data.magbin_zbins_bands[zbin],
            fitting_data.N_zbins_bands[zbin],
            fitting_data.lc_data[zbin],
            fitting_data.filter_info.mag_thresh,
            fitting_data.frac_cat,
            fitting_data.data_sky_area_degsq,
        )

    return phot_loss_multiband_multiz


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
