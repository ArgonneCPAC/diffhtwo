import jax.numpy as jnp
from jax import jit as jjit
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt
from diffstarpop_halpha import diffstarpop_halpha_kern as dpop_halpha


@jjit
def _mse(
    lf_smooth_ms_true: jnp.ndarray,
    lf_smooth_ms_pred: jnp.ndarray,
    lf_q_true: jnp.ndarray,
    lf_q_pred: jnp.ndarray,
) -> jnp.float64:
    """Mean squared error function."""
    return jnp.mean(jnp.power(lf_smooth_ms_true - lf_smooth_ms_pred, 2)) + jnp.mean(
        jnp.power(lf_q_true - lf_q_pred, 2)
    )


@jjit
def _loss_kern(
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
    diffstarpop_params,
    mzr_params,
    spspop_params
):
	halpha_lf_pred = dpop_halpha(
		ran_key,
		t_obs,
		mah_params,
		logmp0,
		t_table,
		ssp_data,
		ssp_halpha_luminosity,
		diffstarpop_params,
		mzr_params,
		spspop_params
	)
    
    return _mse(LF_SF_true, halpha_lf_pred.halpha_L_cgs_smooth_ms, LF_Q_true, halpha_lf_pred.halpha_L_cgs_q)


loss_and_grad_func = jjit(value_and_grad(_loss_kern))

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
    diffstarpop_params,
    mzr_params,
    spspop_params
):
	opt_init, opt_update, get_params = optimizers.adam(step_size)
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
                diffstarpop_params,
                mzr_params,
                spspop_params
            )

        opt_state = opt_update(i, grads, opt_state)
        loss_collector.append(loss)

    loss_arr = np.array(loss_collector)
    theta_best_fit = get_params(opt_state)

    return loss_arr, theta_best_fit



