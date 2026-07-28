import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from ..kernels.fq import get_fq_hm, get_fq_sm
from ..kernels.sfh_rapid_q import get_logsfr_obs
from .plot_sfms import get_leja22_sfms_at_z

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_fq(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    label,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 4.4
    fig, ax = plt.subplots(
        2,
        len(zbins),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    labelsize = 10
    fontsize = 10
    labelsize = 9
    alpha = 0.75
    lw = 1.5

    logsm_arr = np.arange(-10, 14, 0.1)
    for zbin in range(n_z_bins):
        z_med = np.median(zbins[zbin])

        logsfms_leja22 = get_leja22_sfms_at_z(z_med, logsm_arr)
        logsfms_func_at_z = interp1d(logsm_arr, logsfms_leja22, kind="linear")

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))
        ax[0][zbin].set_title(z_min_label + " < z < " + z_max_label)

        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
        ) = get_logsfr_obs(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        """log (sSFR) < -11"""
        # sm
        f_q_fit, logsm_bin_centers_fit = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="lgssfr",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit,
            f_q_fit,
            label="log (sSFR) < -11",
            color="#FFB689",
            lw=lw,
            alpha=alpha,
        )

        # hm
        f_q_fit, logmp_bin_centers_fit = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="lgssfr",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit,
            f_q_fit,
            label="log (sSFR) < -11",
            color="#FFB689",
            lw=lw,
            alpha=alpha,
        )

        # sm
        f_q_fit_cen, logsm_bin_centers_fit_cen = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="lgssfr",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#FFB689",
            lw=lw - 0.5,
            ls="--",
        )

        # hm
        f_q_fit_cen, logmp_bin_centers_fit_cen = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="lgssfr",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#FFB689",
            lw=lw - 0.5,
            ls="--",
        )

        # sm
        f_q_fit_sat, logsm_bin_centers_fit_sat = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="lgssfr",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#FFB689",
            lw=lw - 0.5,
            ls=":",
        )

        # hm
        f_q_fit_sat, logmp_bin_centers_fit_sat = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="lgssfr",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#FFB689",
            lw=lw - 0.5,
            ls=":",
        )

        """Leja+22 SFMS"""
        # sm
        f_q_fit, logsm_bin_centers_fit = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="MS-1dex",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit,
            f_q_fit,
            label="Leja+22 SFMS - 1 dex",
            color="#61C0BF",
            lw=lw,
            alpha=alpha,
        )

        # hm
        f_q_fit, logmp_bin_centers_fit = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="MS-1dex",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit,
            f_q_fit,
            label="Leja+22 SFMS - 1 dex",
            color="#61C0BF",
            lw=lw,
            alpha=alpha,
        )

        # sm
        f_q_fit_cen, logsm_bin_centers_fit_cen = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="MS-1dex",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#61C0BF",
            lw=lw - 0.5,
            ls="--",
        )

        # hm
        f_q_fit_cen, logmp_bin_centers_fit_cen = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="MS-1dex",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#61C0BF",
            lw=lw - 0.5,
            ls="--",
        )

        # sm
        f_q_fit_sat, logsm_bin_centers_fit_sat = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="MS-1dex",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#61C0BF",
            lw=lw - 0.5,
            ls=":",
        )

        # hm
        f_q_fit_sat, logmp_bin_centers_fit_sat = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="MS-1dex",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#61C0BF",
            lw=lw - 0.5,
            ls=":",
        )

        """t_q"""
        # sm
        f_q_fit, logsm_bin_centers_fit = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="t_q",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit,
            f_q_fit,
            label="t$_{q}$ < t$_{obs}$",
            color="#000000",
            lw=lw,
            alpha=alpha,
        )

        # hm
        f_q_fit, logmp_bin_centers_fit = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="all",
            quench_thresh="t_q",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit,
            f_q_fit,
            label="t$_{q}$ < t$_{obs}$",
            color="#000000",
            lw=lw,
            alpha=alpha,
        )

        # sm
        f_q_fit_cen, logsm_bin_centers_fit_cen = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="t_q",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#000000",
            lw=lw - 0.5,
            ls="--",
        )

        # hm
        f_q_fit_cen, logmp_bin_centers_fit_cen = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="cen",
            quench_thresh="t_q",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#000000",
            lw=lw - 0.5,
            ls="--",
        )

        # sm
        f_q_fit_sat, logsm_bin_centers_fit_sat = get_fq_sm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="t_q",
        )
        ax[0][zbin].plot(
            logsm_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#000000",
            lw=lw - 0.5,
            ls=":",
        )

        # hm
        f_q_fit_sat, logmp_bin_centers_fit_sat = get_fq_hm(
            logsm_obs,
            logsfr_obs,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
            logsfms_func_at_z,
            type="sat",
            quench_thresh="t_q",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#000000",
            lw=lw - 0.5,
            ls=":",
        )

        ax[0][zbin].set_xlim(8, 12.2)
        ax[0][zbin].set_xticks([8, 9, 10, 11, 12])

        ax[1][zbin].set_xlim(10, 15)
        ax[1][zbin].set_xticks([10, 11, 12, 13, 14, 15])

        ax[0][zbin].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
        ax[1][zbin].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)
        if zbin != 0:
            ax[0][zbin].set_yticklabels([])
            ax[1][zbin].set_yticklabels([])

        for i in range(0, 2):
            ax[i][zbin].minorticks_on()
            ax[i][zbin].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[i][zbin].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )

    ax[0][0].set_ylabel("quenched fraction", fontsize=fontsize)
    ax[1][0].set_ylabel("quenched fraction", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])

    handles, labels = ax[0][-1].get_legend_handles_labels()
    tot_handle = Line2D([], [], linestyle="solid", color="gray", label="cen+sat")
    cen_handle = Line2D([], [], linestyle="--", color="gray", label="cen")
    sat_handle = Line2D([], [], linestyle=":", color="gray", label="sat")

    leg1 = fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.add_artist(leg1)

    fig.text(
        0.075,
        0.96,
        "Quenching Definition:",
        fontsize=labelsize,
        ha="left",
        va="center",
    )

    fig.legend(
        [tot_handle, cen_handle, sat_handle],
        ["cen+sat", "cen", "sat"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + label + "_f_q.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()
