import jax.numpy as jnp
from diffsky.merging.merging_model import DEFAULT_MERGE_U_PARAMS
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_U_PARAMS
from diffsky.ssp_err_model.defaults import ZERO_SSPERR_U_PARAMS
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax.flatten_util import ravel_pytree


def get_u_theta_from_u_params(
    u_diffstarpop_params=DEFAULT_DIFFSTARPOP_U_PARAMS,
    u_spspop_params=DEFAULT_SPSPOP_U_PARAMS,
    u_ssperrpop_params=ZERO_SSPERR_U_PARAMS,
    u_merging_params=DEFAULT_MERGE_U_PARAMS,
):
    u_diffstarpop_theta, u_diffstarpop_unravel = ravel_pytree(u_diffstarpop_params)
    u_spspop_theta, u_spspop_unravel = ravel_pytree(u_spspop_params)
    u_ssperrpop_theta, u_ssperrpop_unravel = ravel_pytree(u_ssperrpop_params)
    u_merging_theta, u_merging_unravel = ravel_pytree(u_merging_params)

    u_theta = (
        u_diffstarpop_theta,
        u_spspop_theta,
        u_ssperrpop_theta,
        u_merging_theta,
    )

    u_unravel = (
        u_diffstarpop_unravel,
        u_spspop_unravel,
        u_ssperrpop_unravel,
        u_merging_unravel,
    )

    return u_theta, u_unravel


def get_trainable_params(fit_type="all"):
    u_theta_default, u_unravel = get_u_theta_from_u_params(
        u_diffstarpop_params=DEFAULT_DIFFSTARPOP_U_PARAMS,
        u_spspop_params=DEFAULT_SPSPOP_U_PARAMS,
        u_ssperrpop_params=ZERO_SSPERR_U_PARAMS,
        u_merging_params=DEFAULT_MERGE_U_PARAMS,
    )

    zero_trainable = (
        jnp.zeros_like(u_theta_default[0], dtype=bool),  # diffstarpop params
        jnp.zeros_like(u_theta_default[1], dtype=bool),  # spspop params
        jnp.zeros_like(u_theta_default[2], dtype=bool),  # ssperrpop params
        jnp.zeros_like(u_theta_default[3], dtype=bool),  # merging params
    )

    if fit_type == "all":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            jnp.ones_like(u_theta_default[2], dtype=bool),  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "burstpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1].at[:24].set(True),  # allow burstpop params to vary
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "dustpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1]
            .at[24:]
            .set(True),  # spspop params: allow dustpop params to vary
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "spspop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "ssperrpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1],  # spspop params
            jnp.ones_like(u_theta_default[2], dtype=bool),
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "diffstarpop":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            zero_trainable[1],  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "merging":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1],  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "spspop+merging":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "diffstarpop+spspop+merging":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "spspop+diffstarpop":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params, u_unravel

    elif fit_type == "spspop+diffstarpop+merging":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params, u_unravel
