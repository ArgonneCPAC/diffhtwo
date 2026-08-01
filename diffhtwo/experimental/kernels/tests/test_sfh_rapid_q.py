import numpy as np
from diffsky.experimental.kernels.rapid_quenching import DEFAULT_RQ_PARAMS
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from jax import random as jran

from ..lc_phot_kern import multiband_lc_phot_kern
from ..sfh_rapid_q import get_logsfr_obs, update_sfh_with_rapid_q


def test_sfh_rapid_q(fake_subset_ssp_data, feniks_tcurves):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    ran_key = jran.key(0)
    z_min = 0.2
    z_max = 0.5
    num_halos = 100

    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        feniks_tcurves,
    )
    sfh_table = phot_data.sfh_table
    t_table = phot_data.t_table
    t_obs = lc_data.t_obs
    p_merge = phot_data.p_merge
    t_q = 10**phot_data.lg_qt

    sfh_table_updated_with_rapid_q, updated_t_q = update_sfh_with_rapid_q(
        sfh_table,
        t_table,
        t_obs,
        t_q,
        p_merge,
    )

    assert np.isfinite(sfh_table_updated_with_rapid_q).all()
    assert np.isfinite(updated_t_q).all()
    assert (updated_t_q <= t_q).all()

    non_merged_idx = np.where(
        (phot_data.p_merge > 0.05)
        & (phot_data.p_merge < DEFAULT_RQ_PARAMS.rq_p_merge_x0 - 0.05)
    )[0]

    assert (updated_t_q[non_merged_idx] == t_q[non_merged_idx]).all()

    (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
        updated_t_q2,
        lc_data,
        phot_data,
        gal_weight,
    ) = get_logsfr_obs(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        feniks_tcurves,
    )

    assert (updated_t_q == updated_t_q2).all()
