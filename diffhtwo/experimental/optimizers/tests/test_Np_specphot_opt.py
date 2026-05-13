import jax.numpy as jnp
import numpy as np
import pytest
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
    diffstarpop_fields = DEFAULT_PARAM_COLLECTION.diffstarpop_params._fields
    spspop_fields = DEFAULT_PARAM_COLLECTION.spspop_params._fields
    ssperr_fields = DEFAULT_PARAM_COLLECTION.ssperr_params._fields
    merging_fields = DEFAULT_PARAM_COLLECTION.merging_params._fields

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

        # Check all diffstarpop grads are nonzero
        diffstarpop_grads = grads[0]
        diffstarpop_zero_grad_params = []
        assert np.isfinite(
            diffstarpop_grads
        ).all(), "some of the diffstarpop grads are not finite"
        for g in range(0, len(diffstarpop_grads)):
            if diffstarpop_grads[g] == 0.0:
                diffstarpop_zero_grad_params.append(diffstarpop_fields[g])

        assert (
            len(diffstarpop_zero_grad_params) == 0
        ), f"These diffstarpop params have exactly zero grads: {diffstarpop_zero_grad_params}"

        # Check all spspop grads are nonzero
        spspop_grads = grads[1]
        spspop_zero_grad_params = []
        assert np.isfinite(
            spspop_grads
        ).all(), "some of the spspop grads are not finite"

        for g in range(0, len(spspop_grads)):
            if spspop_grads[g] == 0.0:
                spspop_zero_grad_params.append(spspop_fields[g])
        assert (
            len(spspop_zero_grad_params) == 0
        ), f"These spspop params have exactly zero grads: {spspop_zero_grad_params}"

        # Check all ssperr grads are nonzero
        ssperr_grads = grads[2]
        ssperr_zero_grad_params = []
        assert np.isfinite(
            ssperr_grads
        ).all(), "some of the ssperr grads are not finite"

        for g in range(0, len(ssperr_grads)):
            if ssperr_grads[g] == 0.0:
                ssperr_zero_grad_params.append(ssperr_fields[g])
        assert (
            len(ssperr_zero_grad_params) == 0
        ), f"These ssperr params have exactly zero grads: {ssperr_zero_grad_params}"

        # Check all merging grads are nonzero
        merging_grads = grads[3]
        merging_zero_grad_params = []
        assert np.isfinite(
            merging_grads
        ).all(), "some of the merging grads are not finite"

        for g in range(0, len(merging_grads)):
            if merging_grads[g] == 0.0:
                merging_zero_grad_params.append(merging_fields[g])
        assert (
            len(merging_zero_grad_params) == 0
        ), f"These merging params have exactly zero grads: {merging_zero_grad_params}"

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


@pytest.mark.skip(
    reason="This will be enabled when gd_specphot_kern_merging is implemented"
)
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
