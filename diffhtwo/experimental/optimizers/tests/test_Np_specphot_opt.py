import jax.numpy as jnp
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    DEFAULT_PARAM_COLLECTION,
    check_param_collection_is_ok,
)
from jax import random as jran
from jax.example_libraries import optimizers as jax_opt

from ... import param_utils as pu
from ..Np_specphot_opt import (
    _loss_and_grad_phot_kern_multi_z,
    _loss_and_grad_sdss_feniks_hizels,
    fit_N_multi_z,
    fit_sdss_feniks_hizels,
)


def test_all_diffsky_u_param_grads_stay_nonzero_multistep(feniks_multi_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data

    n_steps = 10
    step_size = 0.1
    ran_key = jran.key(0)

    u_theta_init = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    opt_init, opt_update, get_params = jax_opt.adam(step_size)

    other = (
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
    )
    opt_state = opt_init(u_theta_init)

    for i in range(n_steps):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_phot_kern_multi_z(u_theta, *other)

        assert np.isfinite(
            grads[0]
        ).all(), "some of the diffstarpop grads are not finite"
        assert (grads[0] != 0.0).all(), "some of the diffstarpop grads are exactly zero"

        assert np.isfinite(grads[1]).all(), "some of the spspop grads are not finite"
        assert (grads[1] != 0.0).all(), "some of the spspop grads are exactly zero"

        assert np.isfinite(grads[2]).all(), "some of the ssperr grads are not finite"
        assert (grads[2] != 0.0).all(), "some of the ssperr grads are exactly zero"

        assert np.isfinite(grads[3]).all(), "some of the merging grads are not finite"
        assert (grads[3] != 0.0).all(), "some of the merging grads are exactly zero"

        opt_state = opt_update(i, grads, opt_state)


def test_phot_opt(feniks_multi_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data

    ran_key = jran.key(0)

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


def test_specphot_opt(fake_subset_ssp_data, feniks_multi_z_data, hizels):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    emline_wave_table = jnp.array([emline_wave_aa])

    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data

    ran_key = jran.key(0)

    # duplicate feniks data for sdss data
    sdss_meta_data, sdss_fitting_data = feniks_meta_data, feniks_fitting_data

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
