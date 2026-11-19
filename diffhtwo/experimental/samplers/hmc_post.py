import jax.numpy as jnp
from diffsky.experimental.inference import prior, utils
from jax.flatten_util import ravel_pytree

from diffhtwo.experimental.loss_kernels.phot_loss import _loss_phot_kern_2d_multiz
from diffhtwo.experimental.param_utils import get_u_theta_from_param_collection


def flat_logposterior_fn(
    var_uparam_flat,
    diffsky_params,
    sdss_fitting_data,
    feniks_fitting_data,
    var_flat_idx,
    lik_key,
):
    """
    Adapted from code by Natalia Rodriguez

    Combined log-posterior = log-likelihood + log-prior.
    Computes the diffsky transform ``f(*u_coll)`` once and threads the result into both the likelihood (via ``loglikelihood_from_param_coll``)
    and the prior (soft-uniform term + Jacobian diagonal).
    This fusion avoids the duplicate ``f`` call that would occur if we simply added ``flat_loglikelihood_fn + flat_logprior_fn``.
    """

    # \theta* from \theta*_var
    uparam_coll = utils.get_uparam_coll_from_var_uparam_flat(
        var_uparam_flat, diffsky_params
    )
    # \theta from \theta*
    param_coll = prior.f(*uparam_coll)

    # Likelihood(\theta)
    u_theta = get_u_theta_from_param_collection(param_coll)
    sdss_loglik = -_loss_phot_kern_2d_multiz(u_theta, lik_key, sdss_fitting_data)
    feniks_loglik = -_loss_phot_kern_2d_multiz(u_theta, lik_key, feniks_fitting_data)
    loglik = sdss_loglik + feniks_loglik

    # -- So far we did the same as flat_loglikelihood_fn. Now we reuse computations to get the prior.

    # Prior: 2 terms
    # first term
    param_flat, _ = ravel_pytree(param_coll)
    lg_dist_term = jnp.sum(
        prior._soft_uniform_log_prior(
            param_flat[var_flat_idx],
            prior.DEFAULT_LOW_FLAT[var_flat_idx],
            prior.DEFAULT_HIGH_FLAT[var_flat_idx],
        )
    )
    # second term
    log_abs_det = prior._var_jac_logdet(uparam_coll, var_flat_idx)

    return loglik + lg_dist_term + log_abs_det
