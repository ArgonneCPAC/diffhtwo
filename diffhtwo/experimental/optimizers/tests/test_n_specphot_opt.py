import os
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from diffsky.experimental import lightcone_generators as lcg
from diffsky.mass_functions import mc_hosts
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_Params_Diffstarpopfits_mgash,
)
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import load_transmission_curve, retrieve_fake_fsps_data
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ... import param_utils as pu
from ...data_loaders import retrieve_tcurves
from ...defaults import SDSS_AREA_DEG2, SDSS_MAGR_THRESH, SDSS_Z_MAX, SDSS_Z_MIN
from ...utils import zbin_vol
from .. import n_specphot_opt

DIFFSTARPOP_UM_plus_exsitu = DiffstarPop_Params_Diffstarpopfits_mgash["smdpl_dr1"]

u_diffstarpop_theta_default, u_diffstarpop_unravel = ravel_pytree(
    DIFFSTARPOP_UM_plus_exsitu
)
BASE_PATH = Path(__file__).resolve().parent.parent

LH_CENTROIDS_PATH = BASE_PATH / "data/lh_centroids"
SDSS_FILTERS_PATH = BASE_PATH / "data/filters"


@pytest.fixture(scope="module")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


def test_n_specphot_opt(fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    # feniks like
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
    d_centroids = 0.2
    lg_n_thresh = -8
    frac_cat = 1.0

    ran_key = jran.key(0)
    ran_key, n_key = jran.split(ran_key, 2)
    z_idx = 0
    lc_z_min = zbins[z_idx][0]
    lc_z_max = zbins[z_idx][1]

    lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                LH_CENTROIDS_PATH,
                "feniks_lh_centroids_z_"
                + str(lc_z_min)
                + "-"
                + str(lc_z_max)
                + "_test.npy",
            )
        )
    )
    d_centroids = jnp.ones((lh_centroids.shape[0], 1)) * d_centroids

    rng = np.random.default_rng(0)
    lg_n_data = rng.uniform(-17, -4, lh_centroids.shape[0])
    lg_n_err = rng.uniform(0.2, 12, lh_centroids.shape[0])
    lg_n_data_err_lh = np.vstack((lg_n_data, lg_n_err))

    num_halos = 100
    lc_sky_area_degsq = 100
    lgmp_min = 10.0
    lgmp_max = mc_hosts.LGMH_MAX

    tcurves = [
        retrieve_tcurves.MegaCam_uS,
        retrieve_tcurves.HSC_G,
        retrieve_tcurves.HSC_R,
        retrieve_tcurves.HSC_I,
        retrieve_tcurves.HSC_Z,
    ]

    n_z_phot_table = 15
    z_phot_table = 10 ** jnp.linspace(
        np.log10(lc_z_min), np.log10(lc_z_max), n_z_phot_table
    )

    lc_args = (
        ran_key,
        num_halos,
        lc_z_min,
        lc_z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_photdata(*lc_args, cosmo_params=DEFAULT_COSMOLOGY)

    fields = (*lc_data._fields, "lc_vol_mpc3")
    lc_vol_mpc3 = zbin_vol(lc_sky_area_degsq, lc_z_min, lc_z_max, DEFAULT_COSMOLOGY)
    values = (*lc_data, lc_vol_mpc3)
    lc_data = namedtuple(lc_data.__class__.__name__, fields)(*values)

    # test phot loss functions
    phot_loss_args = (
        ran_key,
        lg_n_data_err_lh,
        lg_n_thresh,
        dpwm.DEFAULT_PARAM_COLLECTION,
        lc_data,
        emline_wave_aa,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
    )
    phot_loss = n_specphot_opt.get_phot_loss(*phot_loss_args)

    assert np.isfinite(phot_loss)
    assert phot_loss >= 0

    u_theta_default = pu.get_u_theta_from_param_collection(
        dpwm.DEFAULT_PARAM_COLLECTION
    )

    loss_phot_kern_args = (
        u_theta_default,
        ran_key,
        lg_n_data_err_lh,
        lg_n_thresh,
        lc_data,
        emline_wave_aa,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids,
        d_centroids,
        frac_cat,
    )
    loss_phot_kern = n_specphot_opt._loss_phot_kern(*loss_phot_kern_args)

    assert np.isfinite(loss_phot_kern)
    assert loss_phot_kern >= 0

    lg_n_data_err_lh_multi_z = jnp.array([lg_n_data_err_lh, lg_n_data_err_lh])
    lc_data_multi_z = [lc_data, lc_data]
    lc_data_multi_z = pu.stack_lc_data(lc_data_multi_z)
    lh_centroids_multi_z = jnp.array([lh_centroids, lh_centroids])
    d_centroids_multi_z = jnp.array([d_centroids, d_centroids])

    loss_phot_kern_multi_z_args = (
        u_theta_default,
        ran_key,
        lg_n_data_err_lh_multi_z,
        lg_n_thresh,
        lc_data_multi_z,
        emline_wave_aa,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids_multi_z,
        d_centroids_multi_z,
        frac_cat,
    )

    loss_phot_multi_z = n_specphot_opt._loss_phot_kern_multi_z(
        *loss_phot_kern_multi_z_args
    )

    assert np.isfinite(loss_phot_multi_z).all()
    assert (loss_phot_multi_z >= 0).all()

    # sdss like
    sdss_filters = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    sdss_tcurves = []
    for bn_pat in sdss_filters:
        sdss_tcurve = load_transmission_curve(
            bn_pat=bn_pat + "*", drn=SDSS_FILTERS_PATH
        )
        sdss_tcurves.append(sdss_tcurve)

    sdss_mag_columns = [2]
    sdss_mag_thresh_column = 2
    sdss_mag_thresh = SDSS_MAGR_THRESH
    sdss_frac_cat = 0.8

    sdss_z_phot_table = 10 ** jnp.linspace(
        np.log10(SDSS_Z_MIN), np.log10(SDSS_Z_MAX), n_z_phot_table
    )

    sdss_lc_args = (
        ran_key,
        num_halos,
        SDSS_Z_MIN,
        SDSS_Z_MAX,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        sdss_tcurves,
        sdss_z_phot_table,
    )

    sdss_lc_data = lcg.weighted_lc_photdata(
        *sdss_lc_args, cosmo_params=DEFAULT_COSMOLOGY
    )

    fields = (*sdss_lc_data._fields, "lc_vol_mpc3")
    sdss_lc_vol_mpc3 = zbin_vol(
        lc_sky_area_degsq, SDSS_Z_MIN, SDSS_Z_MAX, DEFAULT_COSMOLOGY
    )
    values = (*sdss_lc_data, sdss_lc_vol_mpc3)
    sdss_lc_data = namedtuple(sdss_lc_data.__class__.__name__, fields)(*values)

    sdss_lh_centroids = jnp.asarray(
        np.load(
            os.path.join(
                LH_CENTROIDS_PATH,
                "sdss_lh_centroids_z_"
                + str(SDSS_Z_MIN)
                + "-"
                + str(SDSS_Z_MAX)
                + "_test.npy",
            )
        )
    )
    sdss_d_centroids = jnp.ones((sdss_lh_centroids.shape[0], 1)) * d_centroids

    sdss_rng = np.random.default_rng(1)
    sdss_lg_n_data = sdss_rng.uniform(-17, -4, sdss_lh_centroids.shape[0])
    sdss_lg_n_err = sdss_rng.uniform(0.2, 12, sdss_lh_centroids.shape[0])
    sdss_lg_n_data_err_lh = np.vstack((sdss_lg_n_data, sdss_lg_n_err))

    sdss_phot_loss_args = (
        u_theta_default,
        ran_key,
        sdss_lg_n_data_err_lh,
        lg_n_thresh,
        sdss_lc_data,
        emline_wave_aa,  # dummy arg
        sdss_mag_columns,
        sdss_mag_thresh_column,
        sdss_mag_thresh,
        sdss_lh_centroids,
        sdss_d_centroids,
        sdss_frac_cat,
    )
    sdss_phot_loss = n_specphot_opt._loss_phot_kern(
        *sdss_phot_loss_args, redshift_as_last_dimension_in_lh=True
    )
    assert np.isfinite(sdss_phot_loss)
    assert sdss_phot_loss >= 0

    # test emline loss functions
    emline_lc_z_min = 0.39
    emline_lc_z_max = 0.41
    emline_lc_sky_area_degsq = 0.1
    emline_z_phot_table = 10 ** np.linspace(
        np.log10(emline_lc_z_min), np.log10(emline_lc_z_max), n_z_phot_table
    )

    emline_lc_args = (
        ran_key,
        num_halos,
        emline_lc_z_min,
        emline_lc_z_max,
        lgmp_min,
        lgmp_max,
        emline_lc_sky_area_degsq,
        ssp_data,
        tcurves,
        emline_z_phot_table,
    )
    emline_lc_data = lcg.weighted_lc_photdata(
        *emline_lc_args, cosmo_params=DEFAULT_COSMOLOGY
    )

    fields = (*emline_lc_data._fields, "lc_vol_mpc3")
    emline_lc_vol_mpc3 = zbin_vol(
        emline_lc_sky_area_degsq, emline_lc_z_min, emline_lc_z_max, DEFAULT_COSMOLOGY
    )
    values = (*emline_lc_data, emline_lc_vol_mpc3)
    emline_lc_data = namedtuple(emline_lc_data.__class__.__name__, fields)(*values)

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
        lg_emline_LF_data,
        lg_emline_Lbin_edges_data,
        lg_n_thresh,
        dpwm.DEFAULT_PARAM_COLLECTION,
        emline_lc_data,
        emline_wave_aa,
    )
    emline_loss = n_specphot_opt.get_emline_loss(*emline_loss_args)

    assert np.isfinite(emline_loss)
    assert emline_loss >= 0

    loss_emline_kern_args = (
        u_theta_default,
        ran_key,
        lg_emline_LF_data,
        lg_emline_Lbin_edges_data,
        lg_n_thresh,
        emline_lc_data,
        emline_wave_aa,
    )

    loss_emline_kern = n_specphot_opt._loss_emline_kern(*loss_emline_kern_args)

    assert np.isfinite(loss_emline_kern)
    assert loss_emline_kern >= 0

    # single line, two redshifts
    emline_lc_data_multi = [[emline_lc_data, emline_lc_data]]
    emline_wave_table = jnp.array([emline_wave_aa])
    lg_emline_LF_data_multi_z = [[lg_emline_LF_data, lg_emline_LF_data]]
    lg_emline_Lbin_edges_data_multi_z = [
        [lg_emline_Lbin_edges_data, lg_emline_Lbin_edges_data]
    ]

    loss_phot_and_emline_multi_z_args = (
        u_theta_default,
        ran_key,
        lg_n_data_err_lh_multi_z,
        lg_n_thresh,
        lc_data_multi_z,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids_multi_z,
        d_centroids_multi_z,
        frac_cat,
        lg_emline_LF_data_multi_z,
        lg_emline_Lbin_edges_data_multi_z,
        emline_lc_data_multi,
        emline_wave_table,
    )

    loss_phot_and_emline_multi_z = n_specphot_opt._loss_phot_and_emline_multi_z(
        *loss_phot_and_emline_multi_z_args
    )

    assert np.isfinite(loss_phot_and_emline_multi_z)
    assert loss_phot_and_emline_multi_z >= 0

    trainable_params = pu.get_trainable_params()
    fit_phot_and_emline_multi_z_args = (
        u_theta_default,
        trainable_params,
        ran_key,
        lg_n_data_err_lh_multi_z,
        lg_n_thresh,
        lc_data_multi_z,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        lh_centroids_multi_z,
        d_centroids_multi_z,
        frac_cat,
        lg_emline_LF_data_multi_z,
        lg_emline_Lbin_edges_data_multi_z,
        emline_lc_data_multi,
        emline_wave_table,
    )
    loss_hist, u_theta_fit = n_specphot_opt.fit_phot_and_emline_multi_z(
        *fit_phot_and_emline_multi_z_args
    )

    assert np.isfinite(loss_hist).all()
    for i in range(0, len(u_theta_fit)):
        assert np.isfinite(u_theta_fit[i]).all()
