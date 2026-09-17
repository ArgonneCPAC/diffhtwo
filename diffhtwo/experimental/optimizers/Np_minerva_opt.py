# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

from functools import partial

import jax.numpy as jnp
from jax import jit as jjit
from jax import lax, value_and_grad
from jax.example_libraries import optimizers as jax_opt

from ..loss_kernels.emline_loss import _loss_emline_kern_multiz
from ..loss_kernels.phot_loss import _loss_phot_kern_2d_multiz

_loss_and_grad_phot_kern_2d_multiz = jjit(value_and_grad(_loss_phot_kern_2d_multiz))
_loss_and_grad_emline_kern_multiz = jjit(value_and_grad(_loss_emline_kern_multiz))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_minerva(
    u_theta_init,
    trainable,
    ran_key,
    minerva_phot,
    minerva_halpha,
    halpha_wave_aa,
    n_steps=2,
    step_size=0.1,
    w_phot=1.0,
    w_halpha=1.0,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init)

    def _opt_update(opt_state, i):
        u_theta = get_params(opt_state)
        loss_phot, grad_phot = _loss_and_grad_phot_kern_2d_multiz(
            u_theta,
            ran_key,
            minerva_phot,
        )

        loss_halpha, grad_halpha = _loss_and_grad_emline_kern_multiz(
            u_theta,
            ran_key,
            halpha_wave_aa,
            minerva_halpha,
        )

        loss_phot = w_phot * loss_phot
        loss_halpha = w_halpha * loss_halpha
        loss = loss_phot + loss_halpha

        grads = tuple(
            w_phot * gp + w_halpha * gh for gp, gh in zip(grad_phot, grad_halpha)
        )
        # set grads for untrainable params to 0.0
        grads = tuple(
            jnp.where(train, grad, 0.0) for grad, train in zip(grads, trainable)
        )

        opt_state = opt_update(i, grads, opt_state)
        return opt_state, (loss, loss_phot, loss_halpha)

    opt_state, (
        loss_hist,
        loss_phot_hist,
        loss_halpha_hist,
    ) = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))

    u_theta_fit = get_params(opt_state)

    return loss_hist, loss_phot_hist, loss_halpha_hist, u_theta_fit
