import jax.numpy as jnp
import numpy as np
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils import spspop_param_utils as spspu
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity import umzr
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ..emline_luminosity_pop import emline_luminosity_func_pop, emline_luminosity_pop
from ..optimizers.emline_luminosity_pop_opt import IDX, fit_emline_luminosity

u_theta_default, u_unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_U_PARAMS)
theta_default, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
emline_wave_aa = 6000


def test_emline_luminosity_opt():
    ran_key = jran.key(0)

    # generate lightcone
    ran_key, lc_key = jran.split(ran_key, 2)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 0.1

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
        emline_wave_aa,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    halpha_L_true = emline_luminosity_pop(*args)
    nhalos = jnp.ones_like(halpha_L_true.emline_L_cgs_q)
    (
        lgL_bin_edges,
        halpha_lf_weighted_q_true,
        halpha_lf_weighted_smooth_ms_true,
        halpha_lf_weighted_bursty_ms_true,
    ) = emline_luminosity_func_pop(halpha_L_true, nhalos)

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
        emline_wave_aa,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        DEFAULT_COSMOLOGY,
        FB,
    )

    loss_hist, u_theta_fit_sub = fit_emline_luminosity(
        *fit_args, n_steps=2, step_size=0.02
    )

    assert np.isfinite(loss_hist).all()
