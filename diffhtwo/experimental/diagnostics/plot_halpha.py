import matplotlib.pyplot as plt

from diffhtwo.experimental.kernels.spec_kern import get_halpha_LF_q_ms_burst


def plot_halpha_ms_q_burst(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    plt_show=True,
):
    alpha = 1
    lw = 2

    xlims = []
    for i in range(0, 4):
        xlims.append(
            (hizels.lg_Lbin_edges[0][i].min(), hizels.lg_Lbin_edges[0][i].max())
        )
    ylim = (-5.5, -1.4)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    # fig.subplots_adjust(hspace=0.2, left=0.065, right=0.98, bottom=0.17, top=0.88)

    for i in range(0, 4):
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
        ) = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,  # dummy arg,
            halpha_wave_aa,
        )
        ax[i].errorbar(
            lgL_bin_centers,
            hizels.lg_phi_data[0][i][0],
            hizels.lg_phi_data[0][i][1],
            color="k",
            fmt="s",
            markersize=5,
            alpha=0.5,
            label="HiZELS",
        )

        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF,
            color="k",
            alpha=alpha,
            label="diffsky",
            lw=lw,
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_burst,
            color="orange",
            alpha=alpha,
            label="mc_is_burst",
            lw=lw,
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_ms,
            color="deepskyblue",
            alpha=alpha,
            label="mc_is_ms",
            lw=lw,
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_q,
            color="darkred",
            alpha=alpha,
            label="mc_is_q",
            lw=lw,
        )

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].set_title(" z = " + str(hizels.z[0][i]), y=0.85)

        ax[i].tick_params(
            axis="both",
            which="both",  # major + minor
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.6,
            labelsize=10,
        )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=14)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=14)
    plt.rcParams["legend.fontsize"] = 8
    ax[-1].legend(loc="lower left", framealpha=0.5)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF" + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    if plt_show:
        plt.show()
    plt.close()
