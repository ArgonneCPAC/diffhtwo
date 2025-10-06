from collections import namedtuple

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.data_loaders import retrieve_fake_fsps_data
from jax import random

from .. import bimodal_sfh_opt as bsfh_opt
from .. import pop_sfh
from ..data_loaders import retrieve_fake_fsps_halpha

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_lgmet = ssp_data.ssp_lgmet
ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr
ssp_halpha_line_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()


def test_bimodal_sfh_opt():
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
    LF_true = LF_SF_true + LF_Q_true

    lgsfr_SF_mean_rand = np.random.uniform(0.8, 1.2)  # true at 1
    frac_SF_rand = np.random.uniform(0, 1)  # true at 0.7
    lgsfr_Q_mean_rand = np.random.uniform(-0.2, 0.2)  # true at 0
    theta_rand = theta(lgsfr_SF_mean_rand, frac_SF_rand, lgsfr_Q_mean_rand)

    losses, grads, theta_fit = bsfh_opt._model_optimization_loop(
        theta_rand,
        LF_true,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
        k_SF,
        k_Q,
        n_steps=10,
        step_size=1e-8,
    )
    assert np.float(losses[-1]) < np.float(losses[0])
    # assert np.allclose(theta_true, theta_fit, atol=1e-1)
