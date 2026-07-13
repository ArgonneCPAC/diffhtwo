import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..kernels.sfh_rapid_q import update_logsfr_obs_with_rapid_q
from ..kernels.spec_kern import get_halpha_LF_q_ms_burst, get_lf_from_linelum

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_halpha(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    alpha = 0.75
    lw = 2

    ylim = (-5.0, -1.0)
    xlim = (39.8, 43.5)

    colors_z = [
        # "#001219",  # deep navy
        "#0a7a80",  # teal
        "#80cca8",  # mint
        "#c8b44a",  # warm gold
        "#c87820",  # amber
    ]
    offsets_z = np.array([0, 0.2, 0.5, 0.8])

    fig, ax = plt.subplots(1, figsize=(4.3, 4), constrained_layout=True)

    for i in range(0, 4):
        _res = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            halpha_wave_aa,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,
            num_halos=num_halos,
        )
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
            lg_halpha_LF_in_situ,
            phot_kern_results,
            spec_kern_results,
            lg_halpha_Lbin_edges,
            lc_data,
        ) = _res

        ax.errorbar(
            lgL_bin_centers,
            hizels.lg_phi_data[0][i][0] + offsets_z[i],
            hizels.lg_phi_data[0][i][1],
            color=colors_z[i],
            fmt="s",
            markersize=5,
            alpha=alpha,
        )

        ax.plot(
            lgL_bin_centers,
            lg_halpha_LF + offsets_z[i],
            color=colors_z[i],
            alpha=alpha,
            label=" z = " + str(hizels.z[0][i]),
            lw=lw,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1,
        labelsize=10,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        labelsize=10,
    )

    handles_z, labels_z = ax.get_legend_handles_labels()
    handles = [
        Line2D([], [], color="k", lw=lw, label="diffsky"),
        ax.errorbar(
            [],
            [],
            yerr=[[0.2], [0.2]],  # vertical error bar
            fmt="s",
            color="k",
            markersize=6,
            linestyle="none",
            lw=lw,
            label="Sobral+13 (HiZELS)",
        ),
    ]
    handles = handles_z + handles
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=10,
    )

    ax.set_xlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=14)
    ax.set_ylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=14)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF" + ".png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()


alpha = 1
lw = 2
ylim = (-5.5, -1.4)
labelsize = 11
fontsize = 14


def plot_halpha_ms_q_burst(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    alpha = 1
    lw = 2

    xlims = []
    for i in range(0, 4):
        xlims.append(
            (
                hizels.lg_Lbin_edges[0][i].min() - 0.1,
                hizels.lg_Lbin_edges[0][i].max() + 0.1,
            )
        )

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))  # leave 8% headroom at top

    for i in range(0, 4):
        _res = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            halpha_wave_aa,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,
            num_halos=num_halos,
        )
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
            lg_halpha_LF_in_situ,
            phot_kern_results,
            spec_kern_results,
            lg_halpha_Lbin_edges,
            lc_data,
        ) = _res

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
            ls="--",
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_ms,
            color="deepskyblue",
            alpha=alpha,
            label="mc_is_ms",
            lw=lw,
            ls="--",
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_q,
            color="darkred",
            alpha=alpha,
            label="mc_is_q",
            lw=lw,
            ls="--",
        )

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].set_title(" z = " + str(hizels.z[0][i]), y=0.85)

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=fontsize)
    handles, labels_ = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0))

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF_q_ms_burst" + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_halpha_ssfr(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    xlims = []
    for i in range(0, 4):
        xlims.append(
            (
                hizels.lg_Lbin_edges[0][i].min() - 0.1,
                hizels.lg_Lbin_edges[0][i].max() + 0.1,
            )
        )

    fig, ax = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

    ssfr_bin_edges = np.array([-12, -11, -10, -9, -8])
    colors = ["#C0394B", "#F07030", "#5BE4FF", "#A78BFA"]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(ssfr_bin_edges, len(colors))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=r"log$_{10}$ (sSFR [yr$^{-1}$])", pad=0.01)

    for i in range(0, 4):
        _res = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            halpha_wave_aa,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,
            num_halos=num_halos,
        )
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
            lg_halpha_LF_in_situ,
            phot_kern_results,
            spec_kern_results,
            lg_halpha_Lbin_edges,
            lc_data,
        ) = _res
        gal_weight = lc_data.cen_weight * lc_data.sat_weight

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
            label="diffsky (total)",
            lw=lw,
        )

        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
        ) = update_logsfr_obs_with_rapid_q(phot_kern_results, lc_data)

        logssfr_obs = logsfr_obs - logsm_obs

        for s in range(0, len(ssfr_bin_edges) - 1):
            ssfr_label = "logssfr_obs: " + str(
                np.round(np.median([ssfr_bin_edges[s], ssfr_bin_edges[s + 1]]), 1)
            )

            sel = (logssfr_obs > ssfr_bin_edges[s]) & (
                logssfr_obs <= ssfr_bin_edges[s + 1]
            )

            lg_halpha_LF_ssfr = get_lf_from_linelum(
                spec_kern_results.linelum_gal[sel],
                gal_weight[sel],
                lg_halpha_Lbin_edges,
                lc_data,
            )
            ax[i].plot(
                lgL_bin_centers,
                lg_halpha_LF_ssfr,
                color=colors[s],
                ls="--",
                alpha=alpha,
                label=ssfr_label,
                lw=lw,
            )

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].set_title(" z = " + str(hizels.z[0][i]), y=0.85)

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=fontsize)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF_ssfr" + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_halpha_sfr(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    xlims = []
    for i in range(0, 4):
        xlims.append(
            (
                hizels.lg_Lbin_edges[0][i].min() - 0.1,
                hizels.lg_Lbin_edges[0][i].max() + 0.1,
            )
        )

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0), constrained_layout=True)

    sfr_bin_edges = np.array([-2, -1, 0, 1, 2])
    colors = ["#C0394B", "#F07030", "#5BE4FF", "#A78BFA"]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(sfr_bin_edges, len(colors))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=r"log$_{10}$ (SFR [M$_{\odot}$$yr^{-1}$])", pad=0.01)

    for i in range(0, 4):
        _res = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            halpha_wave_aa,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,
            num_halos=num_halos,
        )
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
            lg_halpha_LF_in_situ,
            phot_kern_results,
            spec_kern_results,
            lg_halpha_Lbin_edges,
            lc_data,
        ) = _res
        gal_weight = lc_data.cen_weight * lc_data.sat_weight

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
            label="diffsky (total)",
            lw=lw,
        )

        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
        ) = update_logsfr_obs_with_rapid_q(phot_kern_results, lc_data)

        for s in range(0, len(sfr_bin_edges) - 1):
            sfr_label = "logsfr_obs: " + str(
                np.round(np.median([sfr_bin_edges[s], sfr_bin_edges[s + 1]]), 1)
            )
            sel = (logsfr_obs > sfr_bin_edges[s]) & (logsfr_obs <= sfr_bin_edges[s + 1])

            lg_halpha_LF_sfr = get_lf_from_linelum(
                spec_kern_results.linelum_gal[sel],
                gal_weight[sel],
                lg_halpha_Lbin_edges,
                lc_data,
            )
            ax[i].plot(
                lgL_bin_centers,
                lg_halpha_LF_sfr,
                color=colors[s],
                ls="--",
                alpha=alpha,
                label=sfr_label,
                lw=lw,
            )

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].set_title(" z = " + str(hizels.z[0][i]), y=0.85)

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=fontsize)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF_sfr" + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_halpha_sfr_single_z(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    xlims = []
    for i in range(0, 4):
        xlims.append(
            (
                hizels.lg_Lbin_edges[0][i].min() - 0.1,
                hizels.lg_Lbin_edges[0][i].max() + 0.1,
            )
        )

    fig, ax = plt.subplots(1, figsize=(3.5, 3), constrained_layout=True)

    sfr_bin_edges = np.array([-2, -1, 0, 1, 2])
    colors = ["#C0394B", "#F07030", "#5BE4FF", "#A78BFA"]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(sfr_bin_edges, len(colors))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=r"log$_{10}$ (SFR [M$_{\odot}$$yr^{-1}$])", pad=0.01)

    i = 3
    _res = get_halpha_LF_q_ms_burst(
        ran_key,
        param_collection,
        halpha_wave_aa,
        hizels.lg_Lbin_edges[0][i],
        hizels.z[0][i],
        hizels.dz[0][i],
        ssp_data,
        tcurves,
        num_halos=num_halos,
    )
    (
        lgL_bin_centers,
        lg_halpha_LF,
        lg_halpha_LF_q,
        lg_halpha_LF_ms,
        lg_halpha_LF_burst,
        lg_halpha_LF_in_situ,
        phot_kern_results,
        spec_kern_results,
        lg_halpha_Lbin_edges,
        lc_data,
    ) = _res
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    ax.errorbar(
        lgL_bin_centers,
        hizels.lg_phi_data[0][i][0],
        hizels.lg_phi_data[0][i][1],
        color="k",
        fmt="s",
        markersize=5,
        alpha=0.5,
        label="HiZELS",
    )

    ax.plot(
        lgL_bin_centers,
        lg_halpha_LF,
        color="k",
        alpha=alpha,
        label="diffsky (total)",
        lw=lw,
    )

    (
        logsfr_obs,
        logsm_obs,
        logsfr_obs_in_situ,
        logsm_obs_in_situ,
    ) = update_logsfr_obs_with_rapid_q(phot_kern_results, lc_data)

    for s in range(0, len(sfr_bin_edges) - 1):
        sfr_label = "logsfr_obs: " + str(
            np.round(np.median([sfr_bin_edges[s], sfr_bin_edges[s + 1]]), 1)
        )
        sel = (logsfr_obs > sfr_bin_edges[s]) & (logsfr_obs <= sfr_bin_edges[s + 1])

        lg_halpha_LF_sfr = get_lf_from_linelum(
            spec_kern_results.linelum_gal[sel],
            gal_weight[sel],
            lg_halpha_Lbin_edges,
            lc_data,
        )
        ax.plot(
            lgL_bin_centers,
            lg_halpha_LF_sfr,
            color=colors[s],
            ls="--",
            alpha=alpha,
            label=sfr_label,
            lw=lw,
        )

    ax.set_xlim(xlims[i])
    ax.set_ylim(ylim)
    ax.set_title(" z = " + str(hizels.z[0][i]), y=0.85)

    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1,
        labelsize=labelsize,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        labelsize=labelsize,
    )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=fontsize)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF_sfr_single_z" + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_halpha_insitu_exsitu(
    ran_key,
    hizels,
    param_collection,
    ssp_data,
    tcurves,
    halpha_wave_aa,
    model_nickname,
    savedir,
    num_halos=100,
    plt_show=True,
):
    xlims = []
    for i in range(0, 4):
        xlims.append(
            (
                hizels.lg_Lbin_edges[0][i].min() - 0.1,
                hizels.lg_Lbin_edges[0][i].max() + 0.1,
            )
        )

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)

    for i in range(0, 4):
        _res = get_halpha_LF_q_ms_burst(
            ran_key,
            param_collection,
            halpha_wave_aa,
            hizels.lg_Lbin_edges[0][i],
            hizels.z[0][i],
            hizels.dz[0][i],
            ssp_data,
            tcurves,
            num_halos=num_halos,
        )
        (
            lgL_bin_centers,
            lg_halpha_LF,
            lg_halpha_LF_q,
            lg_halpha_LF_ms,
            lg_halpha_LF_burst,
            lg_halpha_LF_in_situ,
            phot_kern_results,
            spec_kern_results,
            lg_halpha_Lbin_edges,
            lc_data,
        ) = _res

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
            label="in+ex-situ",
            lw=lw,
        )
        ax[i].plot(
            lgL_bin_centers,
            lg_halpha_LF_in_situ,
            color="deepskyblue",
            ls="--",
            alpha=alpha,
            label="in-situ",
            lw=lw,
        )

        ax[i].set_xlim(xlims[i])
        ax[i].set_ylim(ylim)
        ax[i].set_title(" z = " + str(hizels.z[0][i]), y=0.85)

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.supylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=fontsize)
    plt.rcParams["legend.fontsize"] = 8
    ax[-1].legend(loc="lower left", framealpha=0.5)

    fig.savefig(
        savedir + "/" + model_nickname + "_halpha_LF_insitu_exsitu" + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()
