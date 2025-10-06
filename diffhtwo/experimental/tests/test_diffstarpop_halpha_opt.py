import numpy as np
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils import spspop_param_utils as spspu
from diffstar.defaults import T_TABLE_MIN
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity import umzr
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ..data_loaders import retrieve_fake_fsps_halpha
from ..diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha_L
from ..diffstarpop_halpha import (
    diffstarpop_halpha_lf_weighted as dpop_halpha_lf_weighted,
)
from ..diffstarpop_halpha_opt import IDX, fit_diffstarpop

# from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
# from diffsky.ssp_err_model import ssp_err_model


u_theta_default, u_unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_U_PARAMS)
theta_default, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_halpha_line_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()


def test_diffstarpop_halpha_opt():
    ran_key = jran.key(0)

    # generate lightcone
    ran_key, lc_key = jran.split(ran_key, 2)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    args = (lc_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    z_obs = lc_halopop["z_obs"]
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    logmp0 = lc_halopop["logmp0"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    scatter_params = DEFAULT_SCATTER_PARAMS
    # ssp_err_pop_params = ssp_err_model.DEFAULT_SSPERR_PARAMS

    ran_key, dpop_halpha_true_key = jran.split(ran_key, 2)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        dpop_halpha_true_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
    )
    halpha_L_true = dpop_halpha_L(*args)

    (
        lgL_bin_edges,
        halpha_lf_weighted_q_true,
        halpha_lf_weighted_smooth_ms_true,
        halpha_lf_weighted_bursty_ms_true,
    ) = dpop_halpha_lf_weighted(halpha_L_true)

    halpha_lf_weighted_composite_true = (
        halpha_lf_weighted_q_true
        + halpha_lf_weighted_smooth_ms_true
        + halpha_lf_weighted_bursty_ms_true
    )

    noise_scale = 0.1
    ran_key, perturb_key = jran.split(ran_key, 2)
    u_theta_perturbed = u_theta_default + noise_scale * jran.normal(
        perturb_key, shape=u_theta_default.shape
    )

    ran_key, dpop_halpha_perturbed_key = jran.split(ran_key, 2)
    fit_args = (
        u_theta_perturbed[IDX],
        halpha_lf_weighted_composite_true,
        dpop_halpha_perturbed_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
    )

    loss_hist, u_theta_fit_sub = fit_diffstarpop(*fit_args, n_steps=200, step_size=0.02)

    u_theta_fit_full = u_theta_default.at[IDX].set(u_theta_fit_sub)
    u_diffstarpop_params_fit = u_unravel_fn(u_theta_fit_full)
    diffstarpop_params_best = get_bounded_diffstarpop_params(u_diffstarpop_params_fit)
    theta_fit, _ = ravel_pytree(diffstarpop_params_best)

    # compare true and fitted in bounded space
    assert np.allclose(theta_default, theta_fit, atol=1)
