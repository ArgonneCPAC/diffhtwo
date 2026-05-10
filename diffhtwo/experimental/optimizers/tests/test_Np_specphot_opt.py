from pathlib import Path

import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    DEFAULT_PARAM_COLLECTION,
    check_param_collection_is_ok,
)
from jax import random as jran

from ... import param_utils as pu
from ...data_loaders import load_feniks, load_hizels
from ...latin_hypercube import lh_utils as lhu
from ..Np_specphot_opt import (
    _loss_and_grad_phot_kern_multi_z,
    _loss_and_grad_sdss_feniks_hizels,
    fit_N_multi_z,
    fit_sdss_feniks_hizels,
)

BASE_PATH = Path(__file__).resolve().parent.parent.parent
FENIKS_DRN = BASE_PATH / "data" / "feniks_test_data"
PHOT = "feniks_phot_selected_for_testing.cat"
ZOUT = "feniks_zout_selected_for_testing.ecsv"

HIZELS_DRN = BASE_PATH / "data" / "hizels"


def test_phot_opt(fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    ran_key = jran.key(0)

    # load feniks test data
    feniks = load_feniks.get_feniks_data(
        FENIKS_DRN,
        ran_key,
        ssp_data,
        phot=PHOT,
        zout=ZOUT,
    )

    z_mins = [0.2, 1.0]
    z_maxs = [1.0, 2.0]

    N_centroids = 200
    num_halos = 100
    feniks_meta_data, feniks_fitting_data = lhu.get_zbins_lh_lc(
        ran_key,
        feniks,
        z_mins,
        z_maxs,
        ssp_data,
        N_centroids,
        num_halos=num_halos,
    )

    u_theta = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)

    loss, grads = _loss_and_grad_phot_kern_multi_z(
        u_theta,
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
    )
    assert np.isfinite(loss)
    for g in range(len(grads)):
        assert np.isfinite(grads[g]).all()
        # assert (grad[g] != 0.0).all()

    trainable_params = pu.get_trainable_params(fit_type="all")
    loss_hist, u_theta_fit = fit_N_multi_z(
        u_theta,
        trainable_params,
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
        n_steps=2,
        step_size=0.1,
    )
    assert np.isfinite(loss_hist).all()
    for u in range(len(u_theta_fit)):
        assert np.isfinite(u_theta_fit[u]).all()

    param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
    assert check_param_collection_is_ok(param_collection_fit)


def test_specphot_opt(fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    emline_wave_table = jnp.array([emline_wave_aa])

    ran_key = jran.key(0)

    # load feniks test data
    feniks = load_feniks.get_feniks_data(
        FENIKS_DRN,
        ran_key,
        ssp_data,
        phot=PHOT,
        zout=ZOUT,
    )
    feniks_tcurves = feniks.filter_info.tcurves

    z_mins = [0.2, 1.0]
    z_maxs = [1.0, 2.0]

    N_centroids = 200
    num_halos = 100
    feniks_meta_data, feniks_fitting_data = lhu.get_zbins_lh_lc(
        ran_key,
        feniks,
        z_mins,
        z_maxs,
        ssp_data,
        N_centroids,
        num_halos=num_halos,
    )
    # duplicate feniks data for sdss data
    sdss_meta_data, sdss_fitting_data = feniks_meta_data, feniks_fitting_data

    # load hizels data
    hizels = load_hizels.get_hizels_data(
        HIZELS_DRN,
        ran_key,
        ssp_data,
        feniks_tcurves,
    )

    u_theta = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    loss, grads = _loss_and_grad_sdss_feniks_hizels(
        u_theta,
        ran_key,
        sdss_meta_data,
        sdss_fitting_data,
        feniks_meta_data,
        feniks_fitting_data,
        hizels,
        emline_wave_table,
    )

    assert np.isfinite(loss)
    for g in range(len(grads)):
        assert np.isfinite(grads[g]).all()
        # assert (grad[g] != 0.0).all()

    trainable_params = pu.get_trainable_params(fit_type="all")
    loss_hist, u_theta_fit = fit_sdss_feniks_hizels(
        u_theta,
        trainable_params,
        ran_key,
        sdss_meta_data,
        sdss_fitting_data,
        feniks_meta_data,
        feniks_fitting_data,
        hizels,
        emline_wave_table,
        n_steps=2,
        step_size=0.1,
    )
    assert np.isfinite(loss_hist).all()
    for u in range(len(u_theta_fit)):
        assert np.isfinite(u_theta_fit[u]).all()

    param_collection_fit = pu.get_param_collection_from_u_theta(u_theta_fit)
    assert check_param_collection_is_ok(param_collection_fit)
