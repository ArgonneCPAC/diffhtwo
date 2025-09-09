# flake8: noqa: E402
""" """
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

import jax.numpy as jnp
from jax import jit as jjit
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt
from diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha


@jjit
def _mse(lf_smooth_ms_true, lf_smooth_ms_pred, lf_q_true, lf_q_pred):
    # 1) Make shapes & dtypes explicit and identical
    lf_smooth_ms_true = jnp.asarray(lf_smooth_ms_true, jnp.float64).reshape(-1)
    lf_smooth_ms_pred = jnp.asarray(lf_smooth_ms_pred, jnp.float64).reshape(
        lf_smooth_ms_true.shape
    )
    lf_q_true = jnp.asarray(lf_q_true, jnp.float64).reshape(-1)
    lf_q_pred = jnp.asarray(lf_q_pred, jnp.float64).reshape(lf_q_true.shape)

    # jax.debug.print(
    #     "shapes(ms_true, ms_pred, q_true, q_pred) = {}, {}, {}, {}",
    #     lf_smooth_ms_true.shape,
    #     lf_smooth_ms_pred.shape,
    #     lf_q_true.shape,
    #     lf_q_pred.shape,
    # )
    # jax.debug.print(
    #     "sizes(ms_true, ms_pred, q_true, q_pred) = {}, {}, {}, {}",
    #     lf_smooth_ms_true.size,
    #     lf_smooth_ms_pred.size,
    #     lf_q_true.size,
    #     lf_q_pred.size,
    # )
    # 2) Compute residuals first (avoid huge intermediates)
    diff_smooth_ms = lf_smooth_ms_true - lf_smooth_ms_pred
    diff_q = lf_q_true - lf_q_pred

    # jax.debug.print(
    #     "any large residuals in smooth_ms = {}", jnp.any(~jnp.isfinite(diff_smooth_ms))
    # )

    # jax.debug.print("any large residuals in q = {}", jnp.any(~jnp.isfinite(diff_q)))

    # 3) Guard non-finite *after* residual is formed (so we don't evaluate logs/squares on bad values)
    diff_smooth_ms = jnp.where(jnp.isfinite(diff_smooth_ms), diff_smooth_ms, 0.0)
    diff_q = jnp.where(jnp.isfinite(diff_q), diff_q, 0.0)

    # 4) Use stable reduction (accumulate in fp64)
    # mse_ms = jnp.mean(jnp.square(diff_smooth_ms), dtype=jnp.float64)
    # mse_q = jnp.mean(jnp.square(diff_q), dtype=jnp.float64)

    return jnp.mean(jnp.square(diff_smooth_ms)) + jnp.mean(jnp.square(diff_q))


@jjit
def _loss_kern(
    theta,
    lf_smooth_ms_true,
    lf_q_true,
    # lf_smooth_ms_pred_mine,
    # lf_q_pred_mine,
    ran_key,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    mzr_params,
    spspop_params,
):
    # jax.debug.print(
    #     "1: _mse Inside _loss_kern BEFORE running diffstarpop_halpha_kern w/ my input pred= {}",
    #     _mse(lf_smooth_ms_true, lf_smooth_ms_pred_mine, lf_q_true, lf_q_pred_mine),
    # )

    halpha_lf_pred = dpop_halpha(
        theta,
        ran_key,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_luminosity,
        mzr_params,
        spspop_params,
    )
    lf_smooth_ms_pred = halpha_lf_pred.halpha_L_cgs_smooth_ms
    lf_q_pred = halpha_lf_pred.halpha_L_cgs_q

    # Both of these have to be printed to make loss=0
    jax.debug.print(
        "jnp.allclose(lf_smooth_ms_true, lf_smooth_ms_pred) = {}",
        jnp.allclose(lf_smooth_ms_true, lf_smooth_ms_pred),
    )
    jax.debug.print(
        "jnp.allclose(lf_q_true, lf_q_pred) = {}", jnp.allclose(lf_q_true, lf_q_pred)
    )

    # jax.debug.print(
    #     "3: _mse Inside _loss_kern AFTER running diffstarpop_halpha_kern using its output pred= {}",
    #     _mse(lf_smooth_ms_true, lf_smooth_ms_pred, lf_q_true, lf_q_pred),
    # )

    return _mse(lf_smooth_ms_true, lf_smooth_ms_pred, lf_q_true, lf_q_pred)


loss_and_grad_func = jjit(value_and_grad(_loss_kern))


@jjit
def fit_diffstarpop(
    theta_init,
    lf_smooth_ms_true,
    lf_q_true,
    ran_key,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    ssp_halpha_luminosity,
    mzr_params,
    spspop_params,
    n_steps=1000,
    step_size=1e-2,
):
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(theta_init)
    theta = get_params(opt_state)
    loss_collector = []

    for i in range(n_steps):
        loss, grads = loss_and_grad_func(
            theta,
            lf_smooth_ms_true,
            lf_q_true,
            ran_key,
            t_obs,
            mah_params,
            logmp0,
            t_table,
            ssp_data,
            ssp_halpha_luminosity,
            mzr_params,
            spspop_params,
        )

        opt_state = opt_update(i, grads, opt_state)
        loss_collector.append(loss)

    loss_arr = jnp.array(loss_collector)
    theta_best_fit = get_params(opt_state)

    return loss_arr, theta_best_fit
