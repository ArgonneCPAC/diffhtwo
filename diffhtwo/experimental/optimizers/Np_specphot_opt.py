# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

from functools import partial

import jax.numpy as jnp
from jax import jit as jjit
from jax import lax, value_and_grad, vmap
from jax.example_libraries import optimizers as jax_opt

from ..loss_kernels.emline_loss import _loss_emline_kern_multi_line_multi_z
from ..loss_kernels.phot_loss import (
    _loss_phot_kern,
    _loss_phot_kern_2d_multiz,
    _loss_phot_kern_multiband_multiz,
)

_L_pk = (
    None,
    None,
    None,
    0,
)
_loss_phot_kern_multi_z = jjit(
    lambda *args, **kwargs: jnp.sum(
        vmap(_loss_phot_kern, in_axes=_L_pk)(*args, **kwargs)
    )
)
_loss_and_grad_phot_kern_multi_z = jjit(value_and_grad(_loss_phot_kern_multi_z))

_loss_and_grad_emline_kern_multi_line_multi_z = jjit(
    value_and_grad(_loss_emline_kern_multi_line_multi_z)
)

_loss_and_grad_phot_kern_multiband_multiz = jjit(
    value_and_grad(_loss_phot_kern_multiband_multiz)
)

_loss_and_grad_phot_kern_2d_multiz = jjit(value_and_grad(_loss_phot_kern_2d_multiz))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_N_multi_z(
    u_theta_init,
    trainable,
    ran_key,
    meta_data,
    fitting_data,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        meta_data,
        fitting_data,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_phot_kern_multi_z(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
        tau = 1.0
        scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_N_phot_2d(
    u_theta_init,
    trainable,
    ran_key,
    fitting_data,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        fitting_data,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_phot_kern_2d_multiz(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        # global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
        # tau = 1.0
        # scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        # grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_N_phot_1d(
    u_theta_init,
    trainable,
    ran_key,
    fitting_data,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        fitting_data,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_phot_kern_multiband_multiz(u_theta, *other)
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        # global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
        # tau = 1.0
        # scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        # grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit


@jjit
def pytree_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    feniks_fitting_data,
    hizels_fitting_data,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss_phot, grad_phot = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            feniks_fitting_data,
        )
        loss_emline, grad_emline = _loss_and_grad_emline_kern_multi_line_multi_z(
            u_theta,
            ran_key,
            hizels_fitting_data,
        )
        w_phot = 1.0 / feniks_fitting_data.nbins
        w_emline = 1.0 / hizels_fitting_data.nbins
        loss = w_phot * loss_phot + w_emline * loss_emline
        grads = tuple(
            w_phot * gp + w_emline * ge for gp, ge in zip(grad_phot, grad_emline)
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        global_norm = pytree_norm(grads)
        tau = 1.0
        scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, loss_phot, loss_emline)

    opt_state, (loss_hist, loss_phot_hist, loss_emline_hist) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)
    return loss_hist, loss_phot_hist, loss_emline_hist, u_theta_fit


@jjit
def _loss_sdss_feniks_hizels(
    u_theta,
    ran_key,
    sdss_meta_data,
    sdss_fitting_data,
    feniks_meta_data,
    feniks_fitting_data,
    hizels_fitting_data,
    fit_sdss=False,
    fit_feniks=False,
    fit_hizels=True,
):
    loss = 0.0

    # sdss
    if fit_sdss:
        sdss_phot_loss = _loss_phot_kern_multi_z(
            u_theta,
            ran_key,
            sdss_meta_data,
            sdss_fitting_data,
        )
        loss += sdss_phot_loss

    # feniks
    if fit_feniks:
        feniks_phot_loss = _loss_phot_kern_multi_z(
            u_theta,
            ran_key,
            feniks_meta_data,
            feniks_fitting_data,
        )
        loss += feniks_phot_loss

    # hizels
    if fit_hizels:
        hizels_emline_multi_line_multi_z_loss_args = (
            u_theta,
            ran_key,
            hizels_fitting_data,
        )
        hizels_emline_loss = _loss_emline_kern_multi_line_multi_z(
            *hizels_emline_multi_line_multi_z_loss_args
        )
        loss += hizels_emline_loss

    return loss


_loss_and_grad_sdss_feniks_hizels = jjit(value_and_grad(_loss_sdss_feniks_hizels))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    sdss_meta_data,
    sdss_fitting_data,
    feniks_meta_data,
    feniks_fitting_data,
    hizels_fitting_data,
    n_steps=2,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    other = (
        ran_key,
        sdss_meta_data,
        sdss_fitting_data,
        feniks_meta_data,
        feniks_fitting_data,
        hizels_fitting_data,
    )

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss, grads = _loss_and_grad_sdss_feniks_hizels(
            u_theta,
            *other,
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
        tau = 1.0
        scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)

    return loss_hist, u_theta_fit
