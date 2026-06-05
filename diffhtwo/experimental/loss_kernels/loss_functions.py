import jax.numpy as jnp
from jax import jit as jjit


@jjit
def mse_w(lg_n_pred, lg_n_target, lg_n_target_err, lg_n_thresh=-10):
    mask = lg_n_target > lg_n_thresh
    nbins = jnp.maximum(jnp.sum(mask), 1)

    resid = (lg_n_pred - lg_n_target) ** 2
    chi2 = resid / lg_n_target_err
    chi2 = jnp.where(mask, chi2, 0.0)

    return jnp.sum(chi2) / nbins


# @jjit
# def poisson_loss(N_pred, N_target, eps=1e-12):
#     N_pred = jnp.clip(N_pred, eps, None)
#     return jnp.sum(N_pred - N_target * jnp.log(N_pred))


@jjit
def poisson_loss(N_pred, N_target, eps=1e-12):
    N_pred = jnp.clip(N_pred, eps, None)
    N_eff = jnp.maximum(jnp.sum(N_target), eps)
    return jnp.sum(N_pred - N_target * jnp.log(N_pred)) / N_eff
