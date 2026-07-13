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


@jjit
def pytree_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_feniks(
    u_theta_init,
    trainable,
    ran_key,
    sdss_fitting_data,
    feniks_fitting_data,
    n_steps=2,
    step_size=1e-2,
    w_sdss=1.0,
    w_feniks=1.0,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss_sdss, grad_sdss = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            sdss_fitting_data,
        )

        loss_feniks, grad_feniks = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            feniks_fitting_data,
        )

        loss_sdss = w_sdss * loss_sdss
        loss_feniks = w_feniks * loss_feniks
        loss = loss_sdss + loss_feniks

        grads = tuple(
            w_sdss * gs + w_feniks * gf for gs, gf in zip(grad_sdss, grad_feniks)
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, loss_sdss, loss_feniks)

    opt_state, (loss_hist, loss_sdss_hist, loss_feniks_hist) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)
    return loss_hist, loss_sdss_hist, loss_feniks_hist, u_theta_fit


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_sdss_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    sdss_fitting_data,
    feniks_fitting_data,
    hizels_fitting_data,
    n_steps=2,
    step_size=1e-2,
    w_sdss=1.0,
    w_feniks=1.0,
    w_hizels=1.0,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss_sdss, grad_sdss = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            sdss_fitting_data,
        )

        loss_feniks, grad_feniks = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            feniks_fitting_data,
        )

        loss_hizels, grad_hizels = _loss_and_grad_emline_kern_multi_line_multi_z(
            u_theta,
            ran_key,
            hizels_fitting_data,
        )

        loss_sdss = w_sdss * loss_sdss
        loss_feniks = w_feniks * loss_feniks
        loss_hizels = w_hizels * loss_hizels
        loss = loss_sdss + loss_feniks + loss_hizels

        grads = tuple(
            w_sdss * gs + w_feniks * gf + w_hizels * gh
            for gs, gf, gh in zip(grad_sdss, grad_feniks, grad_hizels)
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, loss_sdss, loss_feniks, loss_hizels)

    opt_state, (
        loss_hist,
        loss_sdss_hist,
        loss_feniks_hist,
        loss_hizels_hist,
    ) = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))
    u_theta_fit = get_params(opt_state)
    return loss_hist, loss_sdss_hist, loss_feniks_hist, loss_hizels_hist, u_theta_fit


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_feniks_hizels(
    u_theta_init,
    trainable,
    ran_key,
    feniks_fitting_data,
    hizels_fitting_data,
    n_steps=2,
    step_size=1e-2,
    w_feniks=1.0,
    w_hizels=1.0,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss_feniks, grad_feniks = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            feniks_fitting_data,
        )
        loss_hizels, grad_hizels = _loss_and_grad_emline_kern_multi_line_multi_z(
            u_theta,
            ran_key,
            hizels_fitting_data,
        )

        loss_feniks = w_feniks * loss_feniks
        loss_hizels = w_hizels * loss_hizels
        loss = loss_feniks + loss_hizels

        grads = tuple(
            w_feniks * gp + w_hizels * ge for gp, ge in zip(grad_feniks, grad_hizels)
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        # clip gradients
        # global_norm = pytree_norm(grads)
        # tau = 1.0
        # scale = jnp.minimum(1.0, tau / (global_norm + 1e-6))
        # grads = tuple(g * scale for g in grads)

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, loss_feniks, loss_hizels)

    opt_state, (loss_hist, loss_feniks_hist, loss_hizels_hist) = lax.scan(
        _opt_update, opt_state, jnp.arange(n_steps)
    )
    u_theta_fit = get_params(opt_state)
    return loss_hist, loss_feniks_hist, loss_hizels_hist, u_theta_fit
