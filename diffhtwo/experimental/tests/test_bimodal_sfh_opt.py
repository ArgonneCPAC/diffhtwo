from .. import bimodal_sfh_opt as bsfh_opt
from .. import pop_sfh
import numpy as np
from jax import random
import jax.numpy as jnp
from collections import namedtuple
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY


def test_bimodal_sfh_opt():
    ssp_lgmet = jnp.linspace(-5.0, 1.0, 12)
    ssp_lg_age_gyr = jnp.linspace(-4.0, 1.3, 107)
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

    z_obs = 0.5
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs

    k_SF = random.PRNGKey(0)
    k_Q = random.PRNGKey(1)

    lgsfr_SF_mean_true = 1
    frac_SF_true = 0.7
    lgsfr_Q_mean_true = 0
    theta = namedtuple("theta", ["lgsfr_SF_mean", "frac_SF", "lgsfr_Q_mean"])
    theta_true = theta(lgsfr_SF_mean_true, frac_SF_true, lgsfr_Q_mean_true)

    (
        lgL_bin_edges,
        L_halpha_cgs_SF_true,
        L_halpha_cgs_Q_true,
        LF_SF_true,
        LF_Q_true,
        SF_weights_true,
    ) = pop_sfh.pop_bimodal(
        theta_true,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
        k_SF,
        k_Q,
    )

    lgsfr_SF_mean_rand = np.random.uniform(0.8, 1.2)  # true at 1
    frac_SF_rand = np.random.uniform(0, 1)  # true at 0.7
    lgsfr_Q_mean_rand = np.random.uniform(-0.2, 0.2)  # true at 0
    theta_rand = theta(lgsfr_SF_mean_rand, frac_SF_rand, lgsfr_Q_mean_rand)

    _, _, _, LF_SF_rand, LF_Q_rand, _ = pop_sfh.pop_bimodal(
        theta_rand,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
        k_SF,
        k_Q,
    )

    losses, grads, theta_fit = bsfh_opt._model_optimization_loop(
        theta_rand,
        LF_SF_true,
        LF_Q_true,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
        k_SF,
        k_Q,
        n_steps=1000,
        step_size=1e-8,
    )

    assert np.allclose(theta_true, theta_fit, atol=1e-2)
