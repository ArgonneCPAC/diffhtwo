import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from matplotlib.lines import Line2D

from ..kernels.smhm import median_smhm_and_exsitu_frac, median_smhm_q_sf

plt.rc("font", family="serif", serif=["Times New Roman"])

ex_situ_frac_color = "#3E7CB1"


def plot_smhm(
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
    fig_height = 2.2
    fig, ax = plt.subplots(
        1,
        len(zbins),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )

    labelsize = 10
    fontsize = 10
    labelsize = 9
    # alpha = 0.25

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
            logmp_obs_min=10.0,
            logmp_obs_max=15.0,
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

        ax[zbin].set_xlim(11, 15)
        ax[zbin].set_ylim(8, 13)
        ax[zbin].set_xticks([11, 12, 13, 14, 15])
        ax[zbin].set_yticks([8, 9, 10, 11, 12])

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
        savedir + "/" + label + "_smhm_med.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


colors_z = [
    "#1B2A4A",  # z = 0.02–0.2   deep indigo-navy
    "#3D6E8C",  # z = 0.2–0.5    slate teal
    "#6FA287",  # z = 0.5–1.0    muted sage-green
    "#D9A441",  # z = 1.0–1.5    warm ochre/gold
    "#C4432B",  # z = 1.5–2.5    burnt sienna-red
]


def plot_smhm_cen_sat(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    label,
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

    labelsize = 10
    fontsize = 10
    labelsize = 10
    alpha = 0.9

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
            logmp_obs_min=10.0,
            logmp_obs_max=15.0,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        # cen+sat
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median - logmp_bin_centers,
            label=r"$z = $" + z_med,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # cen
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_cen - logmp_bin_centers,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_cen = ascii.read(um_drn + "/" + um_smhm_cen_zname[zbin])
        ax[1].plot(
            um_smhm_cen["Log10(Mpeak/Msun)"],
            um_smhm_cen["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # sat
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sat - logmp_bin_centers,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sat = ascii.read(um_drn + "/" + um_smhm_sat_zname[zbin])
        ax[2].plot(
            um_smhm_sat["Log10(Mpeak/Msun)"],
            um_smhm_sat["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(11, 15)
        ax[i].set_ylim(-3.2, -1.2)
        ax[i].set_xticks([11, 12, 13, 14, 15])
        # ax[i].set_yticks([8, 9, 10, 11, 12])

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
        savedir + "/" + label + "_smhm_med_cen_sat.png",
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
    label,
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

    labelsize = 10
    fontsize = 10
    labelsize = 10
    alpha = 0.9

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
            logmp_obs_min=10.0,
            logmp_obs_max=15.0,
            mag_thresh=None,
            frac_cat=None,
        )

        # all
        ax[0].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median - logmp_bin_centers,
            label=r"$z = $" + z_med,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax[0].plot(
            um_smhm["Log10(Mpeak/Msun)"],
            um_smhm["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1,
            ls="--",
            alpha=alpha,
        )

        # SF
        ax[1].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_sf - logmp_bin_centers,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_sf = ascii.read(um_drn + "/" + um_smhm_sf_zname[zbin])
        ax[1].plot(
            um_smhm_sf["Log10(Mpeak/Msun)"],
            um_smhm_sf["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

        # quiescent
        ax[2].plot(
            logmp_bin_centers,
            logsm_obs_weighted_median_q - logmp_bin_centers,
            color=colors_z[zbin],
            lw=1.5,
            alpha=alpha,
        )
        um_smhm_q = ascii.read(um_drn + "/" + um_smhm_q_zname[zbin])
        ax[2].plot(
            um_smhm_q["Log10(Mpeak/Msun)"],
            um_smhm_q["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.0,
            ls="--",
            alpha=alpha,
        )

    for i in range(0, 3):
        ax[i].set_xlim(11, 15)
        ax[i].set_ylim(-3.2, -1.2)
        ax[i].set_xticks([11, 12, 13, 14, 15])
        # ax[i].set_yticks([8, 9, 10, 11, 12])

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
        savedir + "/" + label + "_smhm_med_q_sf.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm_ratio(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    label,
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

    colors_z = ["#001219", "#0a7a80", "#80cca8", "#c87820", "#9b1d20"]
    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 7.1
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    labelsize = 10
    fontsize = 10
    labelsize = 9
    alpha = 0.8

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))

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
            logmp_obs_min=10.0,
            logmp_obs_max=15.0,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )

        # cen+sat
        smhm_ratio = 10 ** (logsm_obs_weighted_median - logmp_bin_centers)
        ax.plot(
            10**logmp_bin_centers,
            smhm_ratio,
            label=z_min_label + " < z < " + z_max_label,
            color=colors_z[zbin],
            lw=2,
            alpha=alpha,
        )

        um_smhm = ascii.read(um_drn + "/" + um_smhm_zname[zbin])
        ax.plot(
            10 ** um_smhm["Log10(Mpeak/Msun)"],
            10 ** um_smhm["Log10(Median_SM/Mpeak)"],
            color=colors_z[zbin],
            lw=1.5,
            ls="--",
            alpha=alpha,
        )

    ax.set_xscale("log")
    ax.set_xlim(1e10, 1e15)

    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1e-1)

    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1,
        labelsize=labelsize,
    )

    ax.minorticks_on()
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        labelsize=labelsize,
    )

    ax.set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)
    ax.set_ylabel("smhm ratio", fontsize=fontsize)
    handles, labels = ax.get_legend_handles_labels()
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
        savedir + "/" + label + "_smhm_med_ratio.png",
        dpi=400,
    )

    if plt_show:
        plt.show()
    plt.close()


# need to incorporate recent updates in this function
# def plot_smhm_z(
#     ran_key,
#     param_collection,
#     zbins,
#     num_halos,
#     ssp_data,
#     tcurves,
#     data_label,
#     savedir,
#     mag_thresh=None,
#     frac_cat=None,
#     in_situ=False,
#     plt_show=True,
# ):
#     n_z_bins = len(zbins)
#     fig_width = 4.5
#     fig_height = 4
#     fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), constrained_layout=True)

#     labelsize = 10
#     fontsize = 14
#     # alpha = 0.25
#     # colors_z = ["#001219", "#0a7a80", "#80cca8", "#c8b44a", "#c87820", "#9b1d20"]
#     colors_z = [
#         "#4B2D8F",  # 19-3748 Deep Violet
#         "#2055A4",  # 19-4150 Classic Blue
#         "#009473",  # 17-5335 Arcadia
#         "#D4A017",  # 14-0951 Saffron
#         "#E8601C",  # 16-1358 Flame
#         "#9B1B30",  # 19-1757 Chili Pepper
#     ]

#     for zbin in range(n_z_bins):
#         z_min = zbins[zbin][0]
#         z_max = zbins[zbin][1]
#         z_min_label = str(np.round(z_min, 2))
#         z_max_label = str(np.round(z_max, 2))

#         """fit"""
#         lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
#             ran_key,
#             param_collection,
#             z_min,
#             z_max,
#             num_halos,
#             ssp_data,
#             tcurves,
#             mag_thresh=mag_thresh,
#             frac_cat=frac_cat,
#         )
#         if in_situ:
#             (
#                 logmp_bin_centers_fit,
#                 logsm_obs_weighted_mean_fit,
#             ) = get_logsm_obs_weighted_mean(
#                 lc_data.logmp_obs, phot_kern_results.logsm_obs_in_situ, gal_weight
#             )
#         else:
#             (
#                 logmp_bin_centers_fit,
#                 logsm_obs_weighted_mean_fit,
#             ) = get_logsm_obs_weighted_mean(
#                 lc_data.logmp_obs, phot_kern_results.logsm_obs, gal_weight
#             )

#         ax.plot(
#             logmp_bin_centers_fit,
#             logsm_obs_weighted_mean_fit,
#             label=z_min_label + " < z < " + z_max_label,
#             color=colors_z[zbin],
#         )

#     ax.set_xlim(11, 14)
#     ax.set_ylim(8, 12)

#     ax.set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

#     ax.minorticks_on()
#     ax.tick_params(
#         which="major",
#         direction="in",
#         top=True,
#         right=True,
#         length=6,
#         width=1,
#         labelsize=labelsize,
#     )
#     ax.tick_params(
#         which="minor",
#         direction="in",
#         top=True,
#         right=True,
#         length=3,
#         width=0.8,
#         labelsize=labelsize,
#     )

#     if in_situ:
#         ax.set_ylabel(
#             r"<log$_{10}$ (M$_{*, in-situ}$ [M$_{\odot}$])>", fontsize=fontsize
#         )
#     else:
#         ax.set_ylabel(r"<log$_{10}$ (M$_{*}$ [M$_{\odot}$])>", fontsize=fontsize)
#     ax.legend(fontsize=10, loc="lower right")

#     if in_situ:
#         fig.savefig(
#             savedir + "/" + data_label + "_smhm_z_insitu.png",
#             dpi=300,
#         )
#     else:
#         fig.savefig(
#             savedir + "/" + data_label + "_smhm_z.png",
#             dpi=300,
#         )

#     if plt_show:
#         plt.show()
#     plt.close()
