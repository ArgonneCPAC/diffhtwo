from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from jax import grad
import jax.numpy as jnp

from IPython.display import display, clear_output

from tqdm.autonotebook import tqdm

from jax import config

config.update("jax_debug_nans", True)
config.update("jax_debug_infs", True)
config.update("jax_enable_x64", True)


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


def _mseloss(
    theta,
    model,
    LF_SF_true,
    LF_Q_true,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
):
    _, _, LF_SF_pred, LF_Q_pred, _ = model(
        theta, ssp_lgmet, ssp_lg_age_gyr, ssp_halpha_line_luminosity, t_obs
    )
    return _mse(LF_SF_true, LF_SF_pred, LF_Q_true, LF_Q_pred)


def _model_optimization_loop(
    theta,
    model,
    LF_SF_true,
    LF_Q_true,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_halpha_line_luminosity,
    t_obs,
    loss=_mseloss,
    n_steps=1000,
    step_size=1e-9,
):
    dloss = grad(loss)

    losses = []
    grad_lgsfr_SF_mean = []
    grad_frac_SF = []
    grad_lgsfr_Q_mean = []

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.7, hspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Second row: 1 plot spanning all 3 columns
    ax4 = fig.add_subplot(gs[1, :])

    for i in tqdm(range(n_steps)):
        grads = dloss(
            dict(
                lgsfr_SF_mean=theta["lgsfr_SF_mean"],
                frac_SF=theta["frac_SF"],
                lgsfr_Q_mean=theta["lgsfr_Q_mean"],
            ),
            model,
            LF_SF_true,
            LF_Q_true,
            ssp_lgmet,
            ssp_lg_age_gyr,
            ssp_halpha_line_luminosity,
            t_obs,
        )

        grad_lgsfr_SF_mean.append(grads["lgsfr_SF_mean"].item())
        grad_frac_SF.append(grads["frac_SF"].item())
        grad_lgsfr_Q_mean.append(grads["lgsfr_Q_mean"].item())

        theta["lgsfr_SF_mean"] = (
            theta["lgsfr_SF_mean"] - step_size * grads["lgsfr_SF_mean"]
        )
        theta["frac_SF"] = theta["frac_SF"] - step_size * grads["frac_SF"]
        theta["lgsfr_Q_mean"] = (
            theta["lgsfr_Q_mean"] - step_size * grads["lgsfr_Q_mean"]
        )

        losses.append(
            _mseloss(
                dict(
                    lgsfr_SF_mean=theta["lgsfr_SF_mean"],
                    frac_SF=theta["frac_SF"],
                    lgsfr_Q_mean=theta["lgsfr_Q_mean"],
                ),
                model,
                LF_SF_true,
                LF_Q_true,
                ssp_lgmet,
                ssp_lg_age_gyr,
                ssp_halpha_line_luminosity,
                t_obs,
            )
        )

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        ax1.plot(grad_lgsfr_SF_mean)
        ax1.ticklabel_format(style="plain", axis="y")
        ax1.set_ylabel("grad(lgsfr_SF_mean)")

        ax2.plot(grad_frac_SF)
        ax2.ticklabel_format(style="plain", axis="y")
        ax2.set_ylabel("grad(frac_SF)")

        ax3.plot(grad_lgsfr_Q_mean)
        ax3.ticklabel_format(style="plain", axis="y")
        ax3.set_ylabel("grad(lgsfr_Q_mean)")

        ax4.plot(losses)
        ax4.ticklabel_format(style="plain", axis="y")
        ax4.set_xlabel("iterations")
        ax4.set_ylabel("MSE loss")

        # Clear the output cell and display the new figure
        clear_output(wait=True)
        display(fig)

        # time.sleep(0.5) # Pause to see the update
    # fig.savefig(
    #    "./figures/tw_hist_weighted_bimodal_fit_loss_stepsize_"
    #    + str(step_size)
    #    + ".pdf"
    # )
    return losses, theta
