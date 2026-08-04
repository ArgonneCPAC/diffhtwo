import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from matplotlib.lines import Line2D

from ..kernels.lc_phot_kern import multiband_lc_phot_kern
from ..kernels.smhm import (
    _get_logsm_obs_weighted_median,
    median_smhm_and_exsitu_frac,
    median_smhm_q_sf,
)
from ..tab_blue_orange_cmap import make_cmap
from .plot_utils import make_thresholded_reduce_C_function, percentile_norm

plt.rc("font", family="serif", serif=["Times New Roman"])
cmap = make_cmap()
ex_situ_frac_color = "#3E7CB1"

LOGMP_OBS_MIN, LOGMP_OBS_MAX = 10.5, 14.5
LOGSM_OBS_MIN, LOGSM_OBS_MAX = 7.0, 12.5

COLORS_Z = ["#2d0b52", "#0491a1", "#a1d661", "#ee6920", "#8a0f17"]

labelsize = 10
fontsize = 10
labelsize = 10
legendsize = 8
alpha = 0.7


def plot_smhm(
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
    fig_height = 2.2
    fig, ax = plt.subplots(
        1,
        len(zbins),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))
        ax[zbin].set_title(r"$z = $" + z_med)

        (
            logmp_bin_centers,
            logsm_obs_weighted_median,
            logsm_obs_weighted_median_cen_in_situ,
            logsm_obs_weighted_median_cen,
            logsm_obs_weighted_median_sat_in_situ,
            logsm_obs_weighted_median_sat,
            ex_situ_frac_median,
        ) = median_smhm_and_exsitu_frac(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            logmp_obs_min=LOGMP_OBS_MIN,
            logmp_obs_max=LOGMP_OBS_MAX,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        # cen+sat
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median,
            label="cen+sat post-merging",
            color="#000000",
            lw=2,
            alpha=1,
        )

        # cen
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_cen_in_situ,
            label="cen pre-merging",
            color="#FFB689",
            lw=1.5,
            ls=":",
        )
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_cen,
            label="cen post-merging",
            color="#FFB689",
            lw=1.5,
            alpha=0.7,
        )

        # sat
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sat_in_situ,
            label="sat pre-merging",
            color="#61C0BF",
            lw=1.5,
            ls=":",
        )
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sat,
            label="sat post-merging",
            color="#61C0BF",
            lw=1.5,
            alpha=0.7,
        )

        # ex-situ frac
        ax_ex_situ_frac = ax[zbin].twinx()
        ax_ex_situ_frac.plot(
            logmp_bin_centers,
            ex_situ_frac_median,
            color=ex_situ_frac_color,  # C8102E",
            lw=1.0,
            alpha=0.8,
            # ls=":",
        )
        ax_ex_situ_frac.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        ax[zbin].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[zbin].set_ylim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)
        ax[zbin].set_xticks([11, 12, 13, 14])
        ax[zbin].set_yticks([7, 8, 9, 10, 11, 12])

        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=False,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        ax_ex_situ_frac.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            left=False,
            length=6,
            width=1,
            labelsize=labelsize,
            colors=ex_situ_frac_color,
        )

        ax_ex_situ_frac.minorticks_on()
        ax_ex_situ_frac.tick_params(
            which="minor",
            direction="in",
            right=True,
            left=False,
            length=3,
            width=0.8,
            labelsize=labelsize,
            colors=ex_situ_frac_color,
        )

        ax_ex_situ_frac.set_ylim(-0.01, 1)
        ax_ex_situ_frac.spines["right"].set_color(ex_situ_frac_color)

        if zbin == n_z_bins - 1:
            ax_ex_situ_frac.set_ylabel(
                "$\U0001D453_{ex-situ}$", color=ex_situ_frac_color, fontsize=fontsize
            )

        ax[zbin].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(
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

    fig.savefig(
        savedir + "/" + run_label + "_smhm_med.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_hexbin(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    d_mh=0.15,
    cmap=cmap,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 1.9
    fig, ax = plt.subplots(
        1,
        len(zbins),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    gridsize = (70, 60)

    hbs = []
    all_counts = []

    # first pass: plot each panel (norm/cmap set later) and stash the counts
    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))
        ax[zbin].set_title(r"$z = $" + z_med)
        lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
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
        logmp_obs = lc_data.logmp_obs
        logsm_obs = phot_data.logsm_obs

        reduce_C_function = make_thresholded_reduce_C_function(gal_weight)

        hb = ax[zbin].hexbin(
            logmp_obs,
            logsm_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            cmap=cmap,
            gridsize=gridsize,
            extent=(LOGMP_OBS_MIN, LOGMP_OBS_MAX, LOGSM_OBS_MIN, LOGSM_OBS_MAX),
        )
        hbs.append(hb)
        all_counts.append(hb.get_array())

        logmp_bins = np.arange(LOGMP_OBS_MIN, LOGMP_OBS_MAX + d_mh, d_mh)
        logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2
        logsm_obs_weighted_median = _get_logsm_obs_weighted_median(
            logmp_bins, logmp_obs, logsm_obs, gal_weight
        )
        ax[zbin].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median,
            label="median",
            color="#000000",
            lw=2,
            alpha=0.6,
        )

        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=False,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        ax[zbin].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)
        ax[zbin].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[zbin].set_ylim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)
        ax[zbin].set_xticks([11, 12, 13, 14])
        ax[zbin].set_yticks([7, 8, 9, 10, 11, 12])

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)

    # second pass: shared norm from ALL panels' data, applied to every panel
    norm = percentile_norm(all_counts)
    for hb in hbs:
        hb.set_norm(norm)

    fig.colorbar(hbs[-1], ax=ax, label="$\U0001D453$", pad=0.01)
    ax[-1].legend(fontsize=legendsize)

    fig.savefig(
        savedir + "/" + run_label + "_smhm_hexbin.png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_cen_sat(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    um_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    um_smhm_zname = [
        "smhm_med_z0.1.txt",
        "smhm_med_z0.35.txt",
        "smhm_med_z0.75.txt",
        "smhm_med_z1.25.txt",
        "smhm_med_z2.0.txt",
    ]
    um_smhm_cen_zname = [
        "smhm_med_cen_z0.1.txt",
        "smhm_med_cen_z0.35.txt",
        "smhm_med_cen_z0.75.txt",
        "smhm_med_cen_z1.25.txt",
        "smhm_med_cen_z2.0.txt",
    ]
    um_smhm_sat_zname = [
        "smhm_med_sat_z0.1.txt",
        "smhm_med_sat_z0.35.txt",
        "smhm_med_sat_z0.75.txt",
        "smhm_med_sat_z1.25.txt",
        "smhm_med_sat_z2.0.txt",
    ]

    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 3.2
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    ax[0].set_title("all")
    ax[1].set_title("centrals")
    ax[2].set_title("satellites")

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))

        (
            logmp_bin_centers,
            logsm_obs_weighted_median,
            logsm_obs_weighted_median_cen_in_situ,
            logsm_obs_weighted_median_cen,
            logsm_obs_weighted_median_sat_in_situ,
            logsm_obs_weighted_median_sat,
            ex_situ_frac_median,
        ) = median_smhm_and_exsitu_frac(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            logmp_obs_min=LOGMP_OBS_MIN,
            logmp_obs_max=LOGMP_OBS_MAX,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        # cen+sat
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median,
            label=r"$z = $" + z_med,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # cen
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_cen,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_cen = ascii.read(um_drn + "/" + um_smhm_cen_zname[zbin])
        ax[1].plot(
            um_smhm_cen["Log10(Mpeak/Msun)"],
            um_smhm_cen["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # sat
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sat,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sat = ascii.read(um_drn + "/" + um_smhm_sat_zname[zbin])
        ax[2].plot(
            um_smhm_sat["Log10(Mpeak/Msun)"],
            um_smhm_sat["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[i].set_ylim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)
        ax[i].set_xticks([11, 12, 13, 14, 15])

        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        ax[i].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])
    handles, labels = ax[0].get_legend_handles_labels()
    diffsky_handle = Line2D([], [], linestyle="solid", color="gray", label="cen+sat")
    um_handle = Line2D([], [], linestyle="--", color="gray", label="UMachine-DR1")

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

    fig.legend(
        [diffsky_handle, um_handle],
        ["diffsky", "UMachine-DR1"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + run_label + "_smhm_med_cen_sat.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_ratio_cen_sat(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    um_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    um_smhm_zname = [
        "smhm_med_z0.1.txt",
        "smhm_med_z0.35.txt",
        "smhm_med_z0.75.txt",
        "smhm_med_z1.25.txt",
        "smhm_med_z2.0.txt",
    ]
    um_smhm_cen_zname = [
        "smhm_med_cen_z0.1.txt",
        "smhm_med_cen_z0.35.txt",
        "smhm_med_cen_z0.75.txt",
        "smhm_med_cen_z1.25.txt",
        "smhm_med_cen_z2.0.txt",
    ]
    um_smhm_sat_zname = [
        "smhm_med_sat_z0.1.txt",
        "smhm_med_sat_z0.35.txt",
        "smhm_med_sat_z0.75.txt",
        "smhm_med_sat_z1.25.txt",
        "smhm_med_sat_z2.0.txt",
    ]

    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 3.2
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    ax[0].set_title("all")
    ax[1].set_title("centrals")
    ax[2].set_title("satellites")

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))

        (
            logmp_bin_centers,
            logsm_obs_weighted_median,
            logsm_obs_weighted_median_cen_in_situ,
            logsm_obs_weighted_median_cen,
            logsm_obs_weighted_median_sat_in_situ,
            logsm_obs_weighted_median_sat,
            ex_situ_frac_median,
        ) = median_smhm_and_exsitu_frac(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            logmp_obs_min=LOGMP_OBS_MIN,
            logmp_obs_max=LOGMP_OBS_MAX,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        # cen+sat
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median - logmp_bin_centers,
            label=r"$z = $" + z_med,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # cen
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_cen - logmp_bin_centers,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_cen = ascii.read(um_drn + "/" + um_smhm_cen_zname[zbin])
        ax[1].plot(
            um_smhm_cen["Log10(Mpeak/Msun)"],
            um_smhm_cen["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # sat
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sat - logmp_bin_centers,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sat = ascii.read(um_drn + "/" + um_smhm_sat_zname[zbin])
        ax[2].plot(
            um_smhm_sat["Log10(Mpeak/Msun)"],
            um_smhm_sat["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[i].set_ylim(-3.2, -1.2)
        ax[i].set_xticks([11, 12, 13, 14, 15])

        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        ax[i].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$ / M$_{h}$)", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])
    handles, labels = ax[0].get_legend_handles_labels()
    diffsky_handle = Line2D([], [], linestyle="solid", color="gray", label="cen+sat")
    um_handle = Line2D([], [], linestyle="--", color="gray", label="UMachine-DR1")

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

    fig.legend(
        [diffsky_handle, um_handle],
        ["diffsky", "UMachine-DR1"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + run_label + "_smhm_med_cen_sat_ratio.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_q_sf(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    um_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    um_smhm_zname = [
        "smhm_med_z0.1.txt",
        "smhm_med_z0.35.txt",
        "smhm_med_z0.75.txt",
        "smhm_med_z1.25.txt",
        "smhm_med_z2.0.txt",
    ]
    um_smhm_sf_zname = [
        "smhm_med_sf_z0.1.txt",
        "smhm_med_sf_z0.35.txt",
        "smhm_med_sf_z0.75.txt",
        "smhm_med_sf_z1.25.txt",
        "smhm_med_sf_z2.0.txt",
    ]
    um_smhm_q_zname = [
        "smhm_med_q_z0.1.txt",
        "smhm_med_q_z0.35.txt",
        "smhm_med_q_z0.75.txt",
        "smhm_med_q_z1.25.txt",
        "smhm_med_q_z2.0.txt",
    ]

    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 3.2
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    ax[0].set_title("all")
    ax[1].set_title("Star-forming")
    ax[2].set_title("Quiescent")

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))

        (
            logmp_bin_centers,
            logsm_obs_weighted_median,
            logsm_obs_weighted_median_sf,
            logsm_obs_weighted_median_q,
        ) = median_smhm_q_sf(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            logmp_obs_min=LOGMP_OBS_MIN,
            logmp_obs_max=LOGMP_OBS_MAX,
            mag_thresh=None,
            frac_cat=None,
        )

        # all
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median,
            label=r"$z = $" + z_med,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1,
            ls="--",
            alpha=alpha,
        )

        # SF
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sf,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sf = ascii.read(um_drn + "/" + um_smhm_sf_zname[zbin])
        ax[1].plot(
            um_smhm_sf["Log10(Mpeak/Msun)"],
            um_smhm_sf["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # quiescent
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_q,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_q = ascii.read(um_drn + "/" + um_smhm_q_zname[zbin])
        ax[2].plot(
            um_smhm_q["Log10(Mpeak/Msun)"],
            um_smhm_q["Log10(Median_SM/Msun)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[i].set_ylim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)
        ax[i].set_xticks([11, 12, 13, 14, 15])

        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        ax[i].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$)", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])
    handles, labels = ax[0].get_legend_handles_labels()
    diffsky_handle = Line2D([], [], linestyle="solid", color="gray", label="all")
    um_handle = Line2D([], [], linestyle="--", color="gray", label="UMachine-DR1")

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

    fig.legend(
        [diffsky_handle, um_handle],
        ["diffsky", "UMachine-DR1"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + run_label + "_smhm_med_q_sf.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_ratio_q_sf(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    um_drn,
    mag_thresh=None,
    frac_cat=None,
    plt_show=True,
):
    um_smhm_zname = [
        "smhm_med_z0.1.txt",
        "smhm_med_z0.35.txt",
        "smhm_med_z0.75.txt",
        "smhm_med_z1.25.txt",
        "smhm_med_z2.0.txt",
    ]
    um_smhm_sf_zname = [
        "smhm_med_sf_z0.1.txt",
        "smhm_med_sf_z0.35.txt",
        "smhm_med_sf_z0.75.txt",
        "smhm_med_sf_z1.25.txt",
        "smhm_med_sf_z2.0.txt",
    ]
    um_smhm_q_zname = [
        "smhm_med_q_z0.1.txt",
        "smhm_med_q_z0.35.txt",
        "smhm_med_q_z0.75.txt",
        "smhm_med_q_z1.25.txt",
        "smhm_med_q_z2.0.txt",
    ]

    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 3.2
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    ax[0].set_title("all")
    ax[1].set_title("Star-forming")
    ax[2].set_title("Quiescent")

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))

        (
            logmp_bin_centers,
            logsm_obs_weighted_median,
            logsm_obs_weighted_median_sf,
            logsm_obs_weighted_median_q,
        ) = median_smhm_q_sf(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            logmp_obs_min=LOGMP_OBS_MIN,
            logmp_obs_max=LOGMP_OBS_MAX,
            mag_thresh=None,
            frac_cat=None,
        )

        # all
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median - logmp_bin_centers,
            label=r"$z = $" + z_med,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1,
            ls="--",
            alpha=alpha,
        )

        # SF
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sf - logmp_bin_centers,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sf = ascii.read(um_drn + "/" + um_smhm_sf_zname[zbin])
        ax[1].plot(
            um_smhm_sf["Log10(Mpeak/Msun)"],
            um_smhm_sf["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # quiescent
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_q - logmp_bin_centers,
            color=COLORS_Z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_q = ascii.read(um_drn + "/" + um_smhm_q_zname[zbin])
        ax[2].plot(
            um_smhm_q["Log10(Mpeak/Msun)"],
            um_smhm_q["Log10(Median_SM/Mpeak)"],
            color=COLORS_Z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
        ax[i].set_ylim(-3.2, -1.2)
        ax[i].set_xticks([11, 12, 13, 14, 15])

        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

        ax[i].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{*}$ / M$_{h}$)", fontsize=fontsize)

    fig.get_layout_engine().set(rect=[0, 0, 1, 0.94])
    handles, labels = ax[0].get_legend_handles_labels()
    diffsky_handle = Line2D([], [], linestyle="solid", color="gray", label="all")
    um_handle = Line2D([], [], linestyle="--", color="gray", label="UMachine-DR1")

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

    fig.legend(
        [diffsky_handle, um_handle],
        ["diffsky", "UMachine-DR1"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.savefig(
        savedir + "/" + run_label + "_smhm_med_q_sf_ratio.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()
