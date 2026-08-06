import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from ..kernels.fq import get_fq_hm, get_fq_sm
from ..kernels.sfh_rapid_q import get_logsfr_obs
from .plot_sfms import get_leja22_sfms_at_z

plt.rc("font", family="serif", serif=["Times New Roman"])

LOGMP_OBS_MIN, LOGMP_OBS_MAX = 10.5, 14.5
LOGSM_OBS_MIN, LOGSM_OBS_MAX = 8.0, 12.5


def plot_fq(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
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
    labelsize = 10
    alpha = 0.9
    lw = 1.5
    lw_cen_sat = 1.0

    logsm_arr = np.arange(-10, 14, 0.1)
    for zbin in range(n_z_bins):
        z_med = np.median(zbins[zbin])

        logsfms_leja22 = get_leja22_sfms_at_z(z_med, logsm_arr)
        logsfms_func_at_z = interp1d(logsm_arr, logsfms_leja22, kind="linear")

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))
        ax[0, zbin].set_title(r"$z = $" + z_med)

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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
            ls=":",
        )

        """Leja+22 SFMS"""
        # # sm
        # f_q_fit, logsm_bin_centers_fit = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="all",
        #     quench_thresh="MS-1dex",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit,
        #     f_q_fit,
        #     label="Leja+22 SFMS - 1 dex",
        #     color="#FFB689",
        #     lw=lw,
        #     alpha=alpha,
        # )

        # # hm
        # f_q_fit, logmp_bin_centers_fit = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="all",
        #     quench_thresh="MS-1dex",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit,
        #     f_q_fit,
        #     label="Leja+22 SFMS - 1 dex",
        #     color="#FFB689",
        #     lw=lw,
        #     alpha=alpha,
        # )

        # # sm
        # f_q_fit_cen, logsm_bin_centers_fit_cen = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="cen",
        #     quench_thresh="MS-1dex",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit_cen,
        #     f_q_fit_cen,
        #     color="#FFB689",
        #     lw=lw_cen_sat,
        #     ls="--",
        # )

        # # hm
        # f_q_fit_cen, logmp_bin_centers_fit_cen = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="cen",
        #     quench_thresh="MS-1dex",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit_cen,
        #     f_q_fit_cen,
        #     color="#FFB689",
        #     lw=lw_cen_sat,
        #     ls="--",
        # )

        # # sm
        # f_q_fit_sat, logsm_bin_centers_fit_sat = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="sat",
        #     quench_thresh="MS-1dex",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit_sat,
        #     f_q_fit_sat,
        #     color="#FFB689",
        #     lw=lw_cen_sat,
        #     ls=":",
        # )

        # # hm
        # f_q_fit_sat, logmp_bin_centers_fit_sat = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="sat",
        #     quench_thresh="MS-1dex",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit_sat,
        #     f_q_fit_sat,
        #     color="#FFB689",
        #     lw=lw_cen_sat,
        #     ls=":",
        # )

        # """t_q"""
        # # sm
        # f_q_fit, logsm_bin_centers_fit = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="all",
        #     quench_thresh="t_q",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit,
        #     f_q_fit,
        #     label="t$_{q}$ < t$_{obs}$",
        #     color="#61C0BF",
        #     lw=lw,
        #     alpha=alpha,
        # )

        # # hm
        # f_q_fit, logmp_bin_centers_fit = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="all",
        #     quench_thresh="t_q",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit,
        #     f_q_fit,
        #     label="t$_{q}$ < t$_{obs}$",
        #     color="#61C0BF",
        #     lw=lw,
        #     alpha=alpha,
        # )

        # # sm
        # f_q_fit_cen, logsm_bin_centers_fit_cen = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="cen",
        #     quench_thresh="t_q",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit_cen,
        #     f_q_fit_cen,
        #     color="#61C0BF",
        #     lw=lw_cen_sat,
        #     ls="--",
        # )

        # # hm
        # f_q_fit_cen, logmp_bin_centers_fit_cen = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="cen",
        #     quench_thresh="t_q",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit_cen,
        #     f_q_fit_cen,
        #     color="#61C0BF",
        #     lw=lw_cen_sat,
        #     ls="--",
        # )

        # # sm
        # f_q_fit_sat, logsm_bin_centers_fit_sat = get_fq_sm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="sat",
        #     quench_thresh="t_q",
        # )
        # ax[0][zbin].plot(
        #     logsm_bin_centers_fit_sat,
        #     f_q_fit_sat,
        #     color="#61C0BF",
        #     lw=lw_cen_sat,
        #     ls=":",
        # )

        # # hm
        # f_q_fit_sat, logmp_bin_centers_fit_sat = get_fq_hm(
        #     logsm_obs,
        #     logsfr_obs,
        #     t_q,
        #     lc_data,
        #     phot_data,
        #     gal_weight,
        #     logsfms_func_at_z,
        #     type="sat",
        #     quench_thresh="t_q",
        # )
        # ax[1][zbin].plot(
        #     logmp_bin_centers_fit_sat,
        #     f_q_fit_sat,
        #     color="#61C0BF",
        #     lw=lw_cen_sat,
        #     ls=":",
        # )

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
            quench_thresh="t_q",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit,
            f_q_fit,
            label="t$_{q}$ < t$_{obs}$",
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
            quench_thresh="t_q",
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
            quench_thresh="t_q",
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
            quench_thresh="t_q",
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
            quench_thresh="t_q",
        )
        ax[1][zbin].plot(
            logmp_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#61C0BF",
            lw=lw - 0.5,
            ls=":",
        )

        ax[0][zbin].set_xticks([7, 8, 9, 10, 11, 12])
        ax[0][zbin].set_xlim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)

        ax[1][zbin].set_xticks([10, 11, 12, 13, 14, 15])
        ax[1][zbin].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)

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

    ax[0][0].set_ylabel("quenched fraction, $\U0001D453_{q}$", fontsize=fontsize)
    ax[1][0].set_ylabel("quenched fraction, $\U0001D453_{q}$", fontsize=fontsize)

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
        0.125,
        0.96,
        "Quenching Definition:",
        fontsize=labelsize,
        ha="left",
        va="center",
        fontweight="bold",
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
        savedir + "/" + run_label + "_f_q.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_fq_um(
    ran_key,
    param_collection,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    um_fq_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    zbins = np.array(
        [
            [0.09, 0.11],
            [0.34, 0.36],
            [0.74, 0.76],
            [1.24, 1.26],
            [1.99, 2.01],
        ]
    )
    um_z = ["0.911185", "0.744123", "0.571997", "0.445435", "0.334060"]

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
    labelsize = 10
    alpha = 0.8
    lw = 1.5
    lw_cen_sat = 1.0

    logsm_arr = np.arange(-10, 16, 0.1)
    for zbin in range(n_z_bins):
        z_med = np.median(zbins[zbin])

        logsfms_leja22 = get_leja22_sfms_at_z(z_med, logsm_arr)
        logsfms_func_at_z = interp1d(logsm_arr, logsfms_leja22, kind="linear")

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))
        ax[0, zbin].set_title(r"$z = $" + z_med)

        """Literature"""
        # all sm
        Behroozi2019 = ascii.read(
            um_fq_drn + "/qf_a" + um_z[zbin] + ".dat",
            format="commented_header",
            header_start=2,
        )
        ax[0, zbin].plot(
            Behroozi2019["Log10(SM_Center)"],
            Behroozi2019["FQ(SSFR<1e-11)"],
            c="tab:red",
            alpha=alpha,
            label="Behroozi+19 (UMachine-DR1)",
        )
        ax[0, zbin].fill_between(
            Behroozi2019["Log10(SM_Center)"],
            Behroozi2019["FQ(SSFR<1e-11)"] - Behroozi2019["Err-_1"],
            Behroozi2019["FQ(SSFR<1e-11)"] + Behroozi2019["Err+_1"],
            alpha=0.2,
            color="tab:red",
        )

        # cen/sat sm
        Behroozi2019 = ascii.read(
            um_fq_drn + "/qf_groupstats_a" + um_z[zbin] + ".dat",
            format="commented_header",
            header_start=6,
        )
        ax[0, zbin].plot(
            Behroozi2019["Log10(SM_Center)"],
            Behroozi2019["Median_FQ(Centrals)"],
            c="tab:red",
            alpha=alpha,
            lw=lw_cen_sat,
            ls="--",
        )
        ax[0, zbin].plot(
            Behroozi2019["Log10(SM_Center)"],
            Behroozi2019["Median_FQ(Sats)"],
            c="tab:red",
            alpha=alpha,
            lw=lw_cen_sat,
            ls=":",
        )

        # all hm
        Behroozi2019 = ascii.read(
            um_fq_drn + "/qf_hm_a" + um_z[zbin] + ".dat",
            format="commented_header",
            header_start=2,
        )
        ax[1, zbin].plot(
            Behroozi2019["Log10(HM)"],
            Behroozi2019["FQ(True_SSFR<1e-11/yr)"],
            c="tab:red",
            alpha=alpha,
            label="Behroozi+19 (UMachine-DR1)",
        )
        ax[1, zbin].fill_between(
            Behroozi2019["Log10(HM)"],
            Behroozi2019["FQ(True_SSFR<1e-11/yr)"] - Behroozi2019["Err-"],
            Behroozi2019["FQ(True_SSFR<1e-11/yr)"] + Behroozi2019["Err+"],
            alpha=0.2,
            color="tab:red",
        )

        # cen/sat hm
        Behroozi2019 = ascii.read(
            um_fq_drn + "/qf_hm_groupstats_a" + um_z[zbin] + ".dat",
            format="commented_header",
            header_start=6,
        )
        ax[1, zbin].plot(
            Behroozi2019["Log10(HM)"],
            Behroozi2019["FQ(Centrals)"],
            c="tab:red",
            alpha=alpha,
            lw=lw_cen_sat,
            ls="--",
        )
        ax[1, zbin].plot(
            Behroozi2019["Log10(HM)"],
            Behroozi2019["FQ(Sats)"],
            c="tab:red",
            alpha=alpha,
            lw=lw_cen_sat,
            ls=":",
        )

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

        """Diffsky -- log (sSFR) < -11"""
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
            label="This work (Diffsky)",
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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
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
            lw=lw_cen_sat,
            ls=":",
        )

        ax[0][zbin].set_xticks([7, 8, 9, 10, 11, 12])
        ax[0][zbin].set_xlim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)

        ax[1][zbin].set_xticks([10, 11, 12, 13, 14, 15])
        ax[1][zbin].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)

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

    ax[0][0].set_ylabel("quenched fraction, $\U0001D453_{q}$", fontsize=fontsize)
    ax[1][0].set_ylabel("quenched fraction, $\U0001D453_{q}$", fontsize=fontsize)

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
        0.025,
        0.90,
        "Quenching Definition: log (sSFR) < -11",
        fontsize=labelsize,
        ha="left",
        va="center",
        fontweight="bold",
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
        savedir + "/" + run_label + "_f_q_um.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_fq_lit(
    ran_key,
    param_collection,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    fq_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    zbins = np.array(
        [
            [0.2, 0.5],
            [0.5, 1.0],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
        ]
    )
    muzzin13_z = ["0.2z0.5", "0.5z1.0", "1.0z1.5", "1.5z2.0", "2.0z2.5"]

    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 2.5
    fig, ax = plt.subplots(
        1,
        len(zbins),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    labelsize = 10
    fontsize = 10
    labelsize = 10
    alpha = 0.8
    lw = 1.5
    lw_cen_sat = 1.0

    logsm_arr = np.arange(-10, 16, 0.1)
    for zbin in range(n_z_bins):
        z_med = np.median(zbins[zbin])

        logsfms_leja22 = get_leja22_sfms_at_z(z_med, logsm_arr)
        logsfms_func_at_z = interp1d(logsm_arr, logsfms_leja22, kind="linear")

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))
        ax[zbin].set_title(z_min_label + " < z <" + z_max_label)

        """Literature"""
        Muzzin13 = ascii.read(
            fq_drn + "/Muzzin2013/Vmax_Qfraction_" + muzzin13_z[zbin] + ".dat",
            format="commented_header",
            header_start=15,
        )

        ax[zbin].plot(
            Muzzin13["Mstar"],
            Muzzin13["Qfrac"],
            c="deepskyblue",
            alpha=alpha,
            label="Muzzin+13",
        )
        ax[zbin].fill_between(
            Muzzin13["Mstar"],
            Muzzin13["Qfrac"] - Muzzin13["EL_Qfrac"],
            Muzzin13["Qfrac"] + Muzzin13["EU_Qfrac"],
            alpha=0.2,
            color="deepskyblue",
        )

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

        """Diffsky -- log (sSFR) < -11"""
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
        ax[zbin].plot(
            logsm_bin_centers_fit,
            f_q_fit,
            label="This work (Diffsky)",
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
        ax[zbin].plot(
            logsm_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#FFB689",
            lw=lw_cen_sat,
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
        ax[zbin].plot(
            logsm_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#FFB689",
            lw=lw_cen_sat,
            ls=":",
        )

        ax[zbin].set_xticks([7, 8, 9, 10, 11, 12])
        ax[zbin].set_xlim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)

        ax[zbin].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
        if zbin != 0:
            ax[zbin].set_yticklabels([])

        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    ax[0].set_ylabel("quenched fraction, $\U0001D453_{q}$", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])

    handles, labels = ax[-1].get_legend_handles_labels()
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
        0.025,
        0.90,
        "Quenching Definition: UVJ",
        fontsize=labelsize,
        ha="left",
        va="center",
        fontweight="bold",
    )

    fig.legend(
        [tot_handle, cen_handle, sat_handle],
        ["cen+sat", "cen", "sat"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + run_label + "_f_q_lit.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()
