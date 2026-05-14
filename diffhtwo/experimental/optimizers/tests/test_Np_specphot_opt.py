import jax.numpy as jnp
import numpy as np
import pytest
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    DEFAULT_PARAM_COLLECTION,
    check_param_collection_is_ok,
)
from jax.example_libraries import optimizers as jax_opt

from ... import param_utils as pu
from ..Np_specphot_opt import (
    _loss_and_grad_phot_kern_multi_z,
    _loss_and_grad_sdss_feniks_hizels,
    fit_N_multi_z,
    fit_sdss_feniks_hizels,
)


@pytest.fixture(scope="module")
def multistep_grads(ran_key, feniks_multi_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data
    n_steps = 10
    step_size = 0.1

    u_theta_init = pu.get_u_theta_from_param_collection(DEFAULT_PARAM_COLLECTION)
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    other = (
        ran_key,
        feniks_meta_data,
        feniks_fitting_data,
    )
    opt_state = opt_init(u_theta_init)
    multistep_grads = []
    for i in range(n_steps):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_phot_kern_multi_z(u_theta, *other)
        multistep_grads.append(grads)
        opt_state = opt_update(i, grads, opt_state)
    return multistep_grads, n_steps


@pytest.fixture(scope="module")
def diffsky_param_fields():
    diffstarpop_fields = DEFAULT_PARAM_COLLECTION.diffstarpop_params._fields

    spspop_params = DEFAULT_PARAM_COLLECTION.spspop_params
    spspop_fields = (
        spspop_params.burstpop_params.freqburst_params._fields
        + spspop_params.burstpop_params.fburstpop_params._fields
        + spspop_params.burstpop_params.tburstpop_params._fields
        + spspop_params.dustpop_params.avpop_params._fields
        + spspop_params.dustpop_params.deltapop_params._fields
        + spspop_params.dustpop_params.funopop_params._fields
    )

    ssperr_fields = DEFAULT_PARAM_COLLECTION.ssperr_params._fields

    merging_fields = DEFAULT_PARAM_COLLECTION.merging_params._fields

    return diffstarpop_fields, spspop_fields, ssperr_fields, merging_fields


@pytest.fixture(scope="module")
def diffsky_param_names():
    return "diffstarpop", "spspop", "ssperr", "merging"


@pytest.mark.parametrize("param_idx", [0, 1, 2, 3])
def test_all_diffsky_u_param_grads_are_nonzero(
    param_idx, diffsky_param_fields, diffsky_param_names, multistep_grads
):
    multistep_grads, n_steps = multistep_grads
    fields = diffsky_param_fields[param_idx]

    grads = multistep_grads[0][param_idx]
    n_grads = len(grads)
    zero_grad_flags = np.zeros(n_grads)

    for step in range(len(multistep_grads)):
        grads = multistep_grads[step][param_idx]

        assert np.isfinite(
            grads
        ).all(), f"some of the {diffsky_param_names[param_idx]} grads are not finite"
        for g in range(0, n_grads):
            if grads[g] == 0.0:
                zero_grad_flags[g] += 1

    zero_grad_params = []
    for g in range(0, n_grads):
        if zero_grad_flags[g] == n_steps:
            zero_grad_params.append(fields[g])

    assert (
        len(zero_grad_params) == 0
    ), f"These {diffsky_param_names[param_idx]} have exactly zero grads: {zero_grad_params}"


def test_phot_opt(ran_key, feniks_multi_z_data):
    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data

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
def test_specphot_opt(ran_key, fake_subset_ssp_data, feniks_multi_z_data, hizels):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    emline_wave_table = jnp.array([emline_wave_aa])

    feniks_meta_data, feniks_fitting_data = feniks_multi_z_data

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
