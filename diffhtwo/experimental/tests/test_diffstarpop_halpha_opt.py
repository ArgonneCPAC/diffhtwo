import jax.numpy as jnp
import numpy as np
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.param_utils import spspop_param_utils as spspu
from diffstar.defaults import T_TABLE_MIN
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity import umzr
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ..diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha_L
from ..diffstarpop_halpha import (
    diffstarpop_halpha_lf_weighted as dpop_halpha_lf_weighted,
)
from ..diffstarpop_halpha_opt import fit_diffstarpop

# from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
# from diffsky.ssp_err_model import ssp_err_model


theta_default, unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_PARAMS)
IDX = jnp.arange(8, 56, 1)


def test_bimodal_sfh_opt():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    ssp_lgmet = ssp_data.ssp_lgmet
    ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr
    arr = jnp.array(
        [
            18.99350973,
            15.18340321,
            18.59590888,
            14.51430117,
            15.50420145,
            18.89018259,
            14.6840321,
            15.44170473,
            18.62229038,
            18.23519039,
            15.46710684,
            18.09569229,
            18.72593381,
            18.74917029,
            18.84602614,
            18.90889461,
            20.18508488,
            19.64529091,
            20.39642718,
            19.94077289,
            20.36111634,
            20.77151474,
            21.80102465,
            22.86217362,
            24.00485424,
            25.53092415,
            26.12223314,
            29.38864892,
            30.45950998,
            34.16892198,
            39.6796686,
            39.59340563,
            33.72510107,
            29.60391916,
            23.58247219,
            16.62595202,
            13.70698426,
            8.85376548,
            2.83265497,
            2.0991581,
            1.48929108,
            0.96965843,
            0.58217645,
            0.35032972,
            0.22578957,
            0.14377021,
            0.09140591,
        ]
    )
    noise_std = 2
    ssp_halpha_line_luminosity = [
        np.clip(
            arr + np.random.normal(0, noise_std, size=arr.shape), a_min=0, a_max=None
        )
        for _ in range(ssp_lgmet.size)
    ]
    zeros = np.zeros(ssp_lg_age_gyr.size - len(arr))
    ssp_halpha_line_luminosity = [
        np.append(_, zeros) for _ in ssp_halpha_line_luminosity
    ]
    ssp_halpha_line_luminosity = jnp.array(ssp_halpha_line_luminosity)

    ran_key = jran.key(0)

    # generate lightcone
    ran_key, lc_key = jran.split(ran_key, 2)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    args = (lc_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)

    # z_obs = lc_halopop["z_obs"]
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    logmp0 = lc_halopop["logmp0"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    # scatter_params = DEFAULT_SCATTER_PARAMS
    # ssp_err_pop_params = ssp_err_model.DEFAULT_SSPERR_PARAMS

    ran_key, dpop_halpha_true_key = jran.split(ran_key, 2)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        dpop_halpha_true_key,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        mzr_params,
        spspop_params,
    )
    halpha_L_true = dpop_halpha_L(*args)

    (
        lgL_bin_edges,
        halpha_lf_weighted_smooth_ms_true,
        halpha_lf_weighted_q_true,
    ) = dpop_halpha_lf_weighted(halpha_L_true)

    halpha_lf_weighted_composite_true = (
        halpha_lf_weighted_smooth_ms_true + halpha_lf_weighted_q_true
    )

    noise_scale = 0.1
    ran_key, perturb_key = jran.split(ran_key, 2)
    theta_perturbed = theta_default + noise_scale * jran.normal(
        perturb_key, shape=theta_default.shape
    )

    ran_key, dpop_halpha_perturbed_key = jran.split(ran_key, 2)
    fit_args = (
        theta_perturbed[IDX],
        halpha_lf_weighted_composite_true,
        dpop_halpha_perturbed_key,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        mzr_params,
        spspop_params,
    )

    loss_hist, theta_best_fit = fit_diffstarpop(*fit_args, n_steps=200, step_size=1e-3)

    theta_fit = theta_default.at[IDX].set(theta_best_fit)

    assert np.allclose(theta_default, theta_fit, atol=1e-1)
