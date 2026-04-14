import jax.numpy as jnp
from diffsky.experimental.scatter import DEFAULT_SCATTER_U_PARAMS
from diffsky.merging.merging_model import DEFAULT_MERGE_U_PARAMS
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from diffsky.param_utils.spspop_param_utils import DEFAULT_SPSPOP_U_PARAMS
from diffsky.ssp_err_model.defaults import ZERO_SSPERR_U_PARAMS
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from dsps.metallicity.umzr import DEFAULT_MZR_U_PARAMS
from jax import tree_util
from jax.flatten_util import ravel_pytree


def get_u_unravel_fn(
    u_diffstarpop_params=DEFAULT_DIFFSTARPOP_U_PARAMS,
    u_spspop_params=DEFAULT_SPSPOP_U_PARAMS,
    u_ssperrpop_params=ZERO_SSPERR_U_PARAMS,
    u_merging_params=DEFAULT_MERGE_U_PARAMS,
):
    _, u_diffstarpop_unravel = ravel_pytree(u_diffstarpop_params)
    _, u_spspop_unravel = ravel_pytree(u_spspop_params)
    _, u_ssperrpop_unravel = ravel_pytree(u_ssperrpop_params)
    _, u_merging_unravel = ravel_pytree(u_merging_params)

    u_unravel_fn = (
        u_diffstarpop_unravel,
        u_spspop_unravel,
        u_ssperrpop_unravel,
        u_merging_unravel,
    )
    return u_unravel_fn


def get_param_collection_from_u_theta(
    u_theta,
    u_mzr_params=DEFAULT_MZR_U_PARAMS,
    u_scatter_params=DEFAULT_SCATTER_U_PARAMS,
):
    u_unravel_fn = get_u_unravel_fn()

    u_diffstarpop_params = u_unravel_fn[0](u_theta[0])
    u_spspop_params = u_unravel_fn[1](u_theta[1])
    u_ssperrpop_params = u_unravel_fn[2](u_theta[2])
    u_merging_params = u_unravel_fn[3](u_theta[3])

    param_collection = dpwm.get_param_collection_from_u_param_collection(
        u_diffstarpop_params,
        u_mzr_params,
        u_spspop_params,
        u_scatter_params,
        u_ssperrpop_params,
        u_merging_params,
    )

    return param_collection


def get_u_theta_from_u_params(
    u_diffstarpop_params=DEFAULT_DIFFSTARPOP_U_PARAMS,
    u_spspop_params=DEFAULT_SPSPOP_U_PARAMS,
    u_ssperrpop_params=ZERO_SSPERR_U_PARAMS,
    u_merging_params=DEFAULT_MERGE_U_PARAMS,
):
    u_diffstarpop_theta, _ = ravel_pytree(u_diffstarpop_params)
    u_spspop_theta, _ = ravel_pytree(u_spspop_params)
    u_ssperrpop_theta, _ = ravel_pytree(u_ssperrpop_params)
    u_merging_theta, _ = ravel_pytree(u_merging_params)

    u_theta = (
        u_diffstarpop_theta,
        u_spspop_theta,
        u_ssperrpop_theta,
        u_merging_theta,
    )

    return u_theta


def get_u_theta_from_param_collection(param_collection):
    u_param_collection = dpwm.get_u_param_collection_from_param_collection(
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
    )

    u_theta = get_u_theta_from_u_params(
        u_diffstarpop_params=u_param_collection.diffstarpop_u_params,
        u_spspop_params=u_param_collection.spspop_u_params,
        u_ssperrpop_params=u_param_collection.ssperr_u_params,
        u_merging_params=u_param_collection.merging_u_params,
    )

    return u_theta


def get_trainable_params(fit_type="all"):
    u_theta_default = get_u_theta_from_u_params()

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
        return trainable_params

    elif fit_type == "burstpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1].at[:24].set(True),  # allow burstpop params to vary
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params

    elif fit_type == "dustpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1]
            .at[24:]
            .set(True),  # spspop params: allow dustpop params to vary
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params

    elif fit_type == "spspop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params

    elif fit_type == "ssperrpop":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1],  # spspop params
            jnp.ones_like(u_theta_default[2], dtype=bool),
            zero_trainable[3],  # merging params
        )
        return trainable_params

    elif fit_type == "diffstarpop":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            zero_trainable[1],  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params

    elif fit_type == "merging":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            zero_trainable[1],  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params

    elif fit_type == "spspop+merging":
        trainable_params = (
            zero_trainable[0],  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params

    elif fit_type == "diffstarpop+spspop+merging":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            jnp.ones_like(u_theta_default[3], dtype=bool),  # merging params
        )
        return trainable_params

    elif fit_type == "spspop+diffstarpop":
        trainable_params = (
            jnp.ones_like(u_theta_default[0], dtype=bool),  # diffstarpop params
            jnp.ones_like(u_theta_default[1], dtype=bool),  # spspop params
            zero_trainable[2],  # ssperrpop params
            zero_trainable[3],  # merging params
        )
        return trainable_params


def stack_lc_data(lc_data_list):
    treedef = tree_util.tree_structure(lc_data_list[0])
    leaves_list = [tree_util.tree_leaves(lc) for lc in lc_data_list]
    stacked_leaves = [
        jnp.stack([l[i] for l in leaves_list]) for i in range(len(leaves_list[0]))
    ]
    return tree_util.tree_unflatten(treedef, stacked_leaves)
