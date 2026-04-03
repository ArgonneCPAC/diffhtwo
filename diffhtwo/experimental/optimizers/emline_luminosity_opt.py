# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

from functools import partial

import jax.numpy as jnp
from diffstar.diffstarpop import get_bounded_diffstarpop_params
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from jax import jit as jjit
from jax import lax, value_and_grad
from jax.example_libraries import optimizers as jax_opt
from jax.flatten_util import ravel_pytree

from ..emline_luminosity_pop import emline_luminosity_func_pop, emline_luminosity_pop

u_theta_default, u_unravel_fn = ravel_pytree(DEFAULT_DIFFSTARPOP_U_PARAMS)
IDX = jnp.arange(16, 22, 1)


@jjit
def _mse(emline_lf_weighted_composite_true, emline_lf_weighted_composite_pred):
    diff = emline_lf_weighted_composite_true - emline_lf_weighted_composite_pred
    return jnp.mean(jnp.square(diff))


def make_subspace_loss(u_unravel_fn, u_theta_default, IDX):
    """
    Build a loss that optimizes ONLY the parameters at flat indices `IDX`.
    - u_unravel_fn: from ravel_pytree(template)
    - u_theta_default: 1D base vector (others stay fixed to these values)
    - IDX: 1D array/list of flat indices to vary (static for the compiled fn)

    Notes: The only thing you should not do is use namedtuple._replace() / ._asdict()
            inside @jjit. Those are Python-side and will slow/break JIT.
    """
    IDX = jnp.asarray(IDX, dtype=jnp.int64)  # capture in closure

    @jjit
    def _loss_kern_subspace(
        u_theta_sub,  # only the selected subset: shape (len(IDX),)
        emline_lf_weighted_composite_true,
        ran_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        emline_wave_aa,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        cosmo_params,
        fb,
    ):
        # scatter the subset into the full flat vector
        u_theta_full = u_theta_default.at[IDX].set(u_theta_sub)

        # back to structured params and do the usual
        u_diffstarpop_params = u_unravel_fn(u_theta_full)

        # convert to bounded params
        diffstarpop_params = get_bounded_diffstarpop_params(u_diffstarpop_params)

        emline_lf_pred = emline_luminosity_pop(
            diffstarpop_params,
            ran_key,
            z_obs,
            t_obs,
            mah_params,
            logmp0,
            t_table,
            ssp_data,
            emline_wave_aa,
            z_phot_table,
            mzr_params,
            spspop_params,
            scatter_params,
            cosmo_params,
            fb,
        )

        nhalos = jnp.ones_like(emline_lf_pred.emline_L_cgs_q)
        (
            lgL_bin_edges,
            emline_lf_weighted_q_pred,
            emline_lf_weighted_smooth_ms_pred,
            emline_lf_weighted_bursty_ms_pred,
        ) = emline_luminosity_func_pop(emline_lf_pred, nhalos)

        emline_lf_weighted_composite_pred = (
            emline_lf_weighted_q_pred
            + emline_lf_weighted_smooth_ms_pred
            + emline_lf_weighted_bursty_ms_pred
        )

        return _mse(
            emline_lf_weighted_composite_true, emline_lf_weighted_composite_pred
        )

    return _loss_kern_subspace


loss_kern = make_subspace_loss(u_unravel_fn, u_theta_default, IDX)
loss_and_grad_fn = jjit(value_and_grad(loss_kern))


@partial(jjit, static_argnames=["n_steps", "step_size"])
def fit_emline_luminosity(
    u_theta_init_sub,  # only the selected subset: shape (len(IDX),)
    emline_lf_weighted_composite_true,
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    emline_wave_aa,
    z_phot_table,
    mzr_params,
    spspop_params,
    scatter_params,
    cosmo_params,
    fb,
    n_steps=10,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(u_theta_init_sub)

    other = (
        emline_lf_weighted_composite_true,
        ran_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        emline_wave_aa,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        cosmo_params,
        fb,
    )

    def _opt_update(opt_state, i):
        u_theta_sub = get_params(opt_state)
        loss, grads = loss_and_grad_fn(u_theta_sub, *other)
        opt_state = opt_update(i, grads, opt_state)
        return opt_state, loss

    opt_state, loss_hist = lax.scan(_opt_update, opt_state, jnp.arange(n_steps))

    u_theta_fit_sub = get_params(opt_state)

    return loss_hist, u_theta_fit_sub
