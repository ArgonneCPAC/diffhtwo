# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


import jax.numpy as jnp
from jax import grad
from jax import jit as jjit

from . import pop_sfh


@jjit
def _mse(
    LF_SF_true: jnp.ndarray,
    LF_SF_pred: jnp.ndarray,
    LF_Q_true: jnp.ndarray,
    LF_Q_pred: jnp.ndarray,
) -> jnp.float64:
    """Mean squared error function."""
    return jnp.mean(jnp.power(LF_SF_true - LF_SF_pred, 2)) + jnp.mean(
        jnp.power(LF_Q_true - LF_Q_pred, 2)
    )


@jjit
def _mseloss(
    theta,
    LF_SF_true,
    LF_Q_true,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
    k_SF,
    k_Q,
):
    _, _, _, LF_SF_pred, LF_Q_pred, _ = pop_sfh.pop_bimodal(
        theta,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_halpha_line_luminosity,
        t_obs,
        k_SF,
        k_Q,
    )
    return _mse(LF_SF_true, LF_SF_pred, LF_Q_true, LF_Q_pred)


_dmseloss = grad(_mseloss)


def _model_optimization_loop(
    theta,
    LF_SF_true,
    LF_Q_true,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
    k_SF,
    k_Q,
    loss=_mseloss,
    dloss=_dmseloss,
    n_steps=1000,
    step_size=1e-8,
):
    losses = []
    grads = []

    for i in range(n_steps):
        grad = dloss(
            theta,
            LF_SF_true,
            LF_Q_true,
            ssp_lgmet,
            ssp_lg_age_gyr,
            ssp_halpha_line_luminosity,
            t_obs,
            k_SF,
            k_Q,
        )
        grads.append(grad)

        theta._make([t - step_size * g for t, g in zip(theta, grad)])
        # theta = {k: v - step_size * grad[k] for k, v in theta.items()}

        losses.append(
            _mseloss(
                theta,
                LF_SF_true,
                LF_Q_true,
                ssp_lgmet,
                ssp_lg_age_gyr,
                ssp_halpha_line_luminosity,
                t_obs,
                k_SF,
                k_Q,
            )
        )

    return losses, grads, theta
