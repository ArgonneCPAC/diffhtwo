import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from diffsky.experimental import lc_phot_kern
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils.spspop_param_utils import (
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
)
from diffsky.ssp_err_model.defaults import (
    ZERO_SSPERR_PARAMS,
    ZERO_SSPERR_U_PARAMS,
)
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity.umzr import DEFAULT_MZR_PARAMS
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ..data_loaders import retrieve_tcurves
from ..optimizers import phot_and_emline_opt
from ..utils import zbin_volume

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DEFAULT_DIFFSTARPOP_PARAMS
)
BASE_PATH = Path(__file__).resolve().parent.parent
LH_CENTROIDS_PATH = BASE_PATH / "data_loaders/data"


@pytest.fixture(scope="module")
def ssp_data():
    return retrieve_fake_fsps_data.load_fake_ssp_data()


def test_phot_and_emline_opt(ssp_data):
    zbins = np.array(
        [
            [0.2, 0.5],
            [1.5, 1.75],
            [2.75, 3.5],
        ]
    )

    mag_columns = [3]
    mag_thresh_column = 3
    mag_thresh = 24.5
    dmag = 0.2
    lg_n_thresh = -8
    frac_cat = 1.0

    ran_key = jran.key(0)
    ran_key, n_key = jran.split(ran_key, 2)
    z_idx = 0
    n_z_phot_table = 15

    lc_z_min = zbins[z_idx][0]
    lc_z_max = zbins[z_idx][1]
    lc_sky_area_degsq = 0.1
    lc_vol_mpc3 = zbin_volume(lc_sky_area_degsq, zlow=lc_z_min, zhigh=lc_z_max).value

    tcurves = [
        retrieve_tcurves.MegaCam_uS,
        retrieve_tcurves.HSC_G,
        retrieve_tcurves.HSC_R,
        retrieve_tcurves.HSC_I,
        retrieve_tcurves.HSC_Z,
    ]

    lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                LH_CENTROIDS_PATH,
                "lh_centroids_z_" + str(lc_z_min) + "-" + str(lc_z_max) + "_test.npy",
            )
        )
    )
    dmag_centroids = jnp.ones((lh_centroids.shape[0], 1)) * dmag

    rng = np.random.default_rng(0)
    lg_n_data = rng.uniform(-17, -4, lh_centroids.shape[0])
    lg_n_err = rng.uniform(0.2, 12, lh_centroids.shape[0])
    lg_n_data_err_lh = np.vstack((lg_n_data, lg_n_err))

    # t_table
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = jnp.log10(t_0)
    t_table = jnp.linspace(T_TABLE_MIN, 10**lgt0, 100)

    z_phot_table = jnp.linspace(lc_z_min, lc_z_max, n_z_phot_table)
    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, DEFAULT_COSMOLOGY
    )
    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    # test phot loss functions
    phot_loss_args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_SPSPOP_PARAMS,
        ZERO_SSPERR_PARAMS,
        lg_n_data_err_lh,
        lg_n_thresh,
        ran_key,
        DEFAULT_MZR_PARAMS,
        DEFAULT_SCATTER_PARAMS,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min,
        lc_z_max,
        lc_sky_area_degsq,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        DEFAULT_COSMOLOGY,
        FB,
        frac_cat,
    )

    phot_loss = phot_and_emline_opt.get_phot_loss(*phot_loss_args)
    assert np.isfinite(phot_loss)

    u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
        DEFAULT_DIFFSTARPOP_U_PARAMS
    )
    u_spspop_theta_default, u_spspop_unravel = ravel_pytree(DEFAULT_SPSPOP_U_PARAMS)

    u_zero_ssperrpop_theta, u_zero_ssperrpop_unravel = ravel_pytree(
        ZERO_SSPERR_U_PARAMS
    )

    u_theta_default = (
        u_diffstarpop_theta_default,
        u_spspop_theta_default,
        u_zero_ssperrpop_theta,
    )
    loss_args = (
        u_theta_default,
        lg_n_data_err_lh,
        lg_n_thresh,
        ran_key,
        DEFAULT_MZR_PARAMS,
        DEFAULT_SCATTER_PARAMS,
        lh_centroids,
        dmag_centroids,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min,
        lc_z_max,
        lc_sky_area_degsq,
        lc_vol_mpc3,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        DEFAULT_COSMOLOGY,
        FB,
        frac_cat,
    )

    phot_loss_kern = phot_and_emline_opt._loss_phot_kern(
        *loss_args,
    )
    assert np.isfinite(phot_loss_kern)

    # test emline loss functions
    emline_wave_aa = 6000
    emline_lc_z_min = 0.39
    emline_lc_z_max = 0.41
    emline_lc_sky_area_degsq = 0.1
    emline_lc_vol_mpc3 = zbin_volume(
        emline_lc_sky_area_degsq, zlow=emline_lc_z_min, zhigh=emline_lc_z_max
    ).value
    lg_emline_LF_data = jnp.array(
        [
            [
                -1.70275854,
                -1.74275854,
                -1.85275854,
                -1.97275854,
                -2.00275854,
                -2.07275854,
                -2.16275854,
                -2.31275854,
                -2.33275854,
                -2.46275854,
                -2.50275854,
                -2.61275854,
                -2.73275854,
                -2.77275854,
                -2.92275854,
                -3.07275854,
                -3.60275854,
                -3.75275854,
            ],
            [
                0.04,
                0.04,
                0.04,
                0.05,
                0.07,
                0.07,
                0.09,
                0.08,
                0.09,
                0.1,
                0.11,
                0.13,
                0.19,
                0.17,
                0.2,
                0.35,
                0.51,
                0.71,
            ],
        ]
    )

    lg_emline_Lbin_edges_data = jnp.linspace(40, 42.5, lg_emline_LF_data.shape[1] + 1)
    emline_loss_args = (
        ran_key,
        emline_wave_aa,
        lg_emline_LF_data,
        lg_emline_Lbin_edges_data,
        lg_n_thresh,
        emline_lc_z_min,
        emline_lc_z_max,
        emline_lc_sky_area_degsq,
        emline_lc_vol_mpc3,
        t_table,
        ssp_data,
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_SPSPOP_PARAMS,
        DEFAULT_MZR_PARAMS,
        DEFAULT_SCATTER_PARAMS,
        DEFAULT_COSMOLOGY,
        FB,
    )
    emline_loss = phot_and_emline_opt.get_emline_loss(*emline_loss_args)

    assert np.isfinite(emline_loss)

    emline_loss_kern = phot_and_emline_opt._loss_emline_kern(
        u_theta_default,
        ran_key,
        emline_wave_aa,
        lg_emline_LF_data,
        lg_emline_Lbin_edges_data,
        lg_n_thresh,
        emline_lc_z_min,
        emline_lc_z_max,
        emline_lc_sky_area_degsq,
        emline_lc_vol_mpc3,
        t_table,
        ssp_data,
        DEFAULT_MZR_PARAMS,
        DEFAULT_SCATTER_PARAMS,
        DEFAULT_COSMOLOGY,
        FB,
    )
    assert np.isfinite(emline_loss_kern)

    # test multi-z loss
    ran_key, n_key = jran.split(ran_key, 2)

    # phot args
    lg_n_data_err_lh_multi_z = jnp.stack([lg_n_data_err_lh, lg_n_data_err_lh], axis=0)
    lh_centroids_multi_z = jnp.stack([lh_centroids, lh_centroids], axis=0)
    dmag_centroids_multi_z = jnp.stack([dmag_centroids, dmag_centroids], axis=0)
    lc_z_min_multi_z = jnp.array([lc_z_min, lc_z_min])
    lc_z_max_multi_z = jnp.array([lc_z_max, lc_z_max])
    lc_sky_area_degsq_multi_z = jnp.array([lc_sky_area_degsq, lc_sky_area_degsq])
    lc_vol_mpc3_multi_z = jnp.array([lc_vol_mpc3, lc_vol_mpc3])
    precomputed_ssp_mag_table_multi_z = jnp.stack(
        [precomputed_ssp_mag_table, precomputed_ssp_mag_table], axis=0
    )
    z_phot_table_multi_z = jnp.stack([z_phot_table, z_phot_table], axis=0)
    wave_eff_table_multi_z = jnp.stack([wave_eff_table, wave_eff_table], axis=0)

    # emline args
    emline_wave_aa = [emline_wave_aa]
    lg_emline_LF_data_multi_z = [
        jnp.stack([lg_emline_LF_data, lg_emline_LF_data], axis=0)
    ]
    lg_emline_Lbin_edges_data_multi_z = [
        jnp.stack([lg_emline_Lbin_edges_data, lg_emline_Lbin_edges_data], axis=0)
    ]
    emline_lc_z_min_multi_z = [jnp.array([0.39, 0.83])]
    emline_lc_z_max_multi_z = [jnp.array([0.41, 0.85])]
    emline_lc_sky_area_degsq_multi_z = [
        jnp.array([emline_lc_sky_area_degsq, emline_lc_sky_area_degsq])
    ]
    emline_lc_vol_mpc3_multi_z = [
        jnp.array(
            [
                zbin_volume(
                    emline_lc_sky_area_degsq,
                    zlow=emline_lc_z_min_multi_z[0],
                    zhigh=emline_lc_z_max_multi_z[0],
                ).value,
                zbin_volume(
                    emline_lc_sky_area_degsq,
                    zlow=emline_lc_z_min_multi_z[1],
                    zhigh=emline_lc_z_max_multi_z[1],
                ).value,
            ]
        )
    ]

    args = (
        lg_n_data_err_lh_multi_z,
        lg_n_thresh,
        n_key,
        DEFAULT_MZR_PARAMS,
        DEFAULT_SCATTER_PARAMS,
        lh_centroids_multi_z,
        dmag_centroids_multi_z,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lc_z_min_multi_z,
        lc_z_max_multi_z,
        lc_sky_area_degsq_multi_z,
        lc_vol_mpc3_multi_z,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table_multi_z,
        z_phot_table_multi_z,
        wave_eff_table_multi_z,
        DEFAULT_COSMOLOGY,
        FB,
        frac_cat,
        emline_wave_aa,
        lg_emline_LF_data_multi_z,
        lg_emline_Lbin_edges_data_multi_z,
        emline_lc_z_min_multi_z,
        emline_lc_z_max_multi_z,
        emline_lc_sky_area_degsq_multi_z,
        emline_lc_vol_mpc3_multi_z,
    )

    loss_phot_and_emline_multi_z = phot_and_emline_opt._loss_phot_and_emline_multi_z(
        u_theta_default, *args
    )
    assert np.isfinite(loss_phot_and_emline_multi_z)

    trainable = (
        jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
        jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
        jnp.ones_like(u_theta_default[2], dtype=bool),  # ssperrpop params
    )

    loss_hist, u_theta_fit = phot_and_emline_opt.fit_phot_and_emline_multi_z(
        u_theta_default,
        trainable,
        *args,
        n_steps=2,
        step_size=1e-2,
    )
    assert np.isfinite(loss_hist).all()
    for i in range(0, len(u_theta_fit)):
        assert np.isfinite(u_theta_fit[i]).all()
