import matplotlib.pyplot as plt
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def get_logsm_obs_weighted_mean(logmp_obs, logsm_obs, gal_weight):
    logmp_bins = np.arange(logmp_obs.min(), logmp_obs.max() + 0.25, 0.25)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    logsm_obs_weighted_mean = []
    for b in range(0, len(logmp_bins) - 1):
        in_bin = (logmp_obs > logmp_bins[b]) & (logmp_obs < logmp_bins[b + 1])
        logsm_obs_weighted_mean.append(
            np.sum(logsm_obs[in_bin] * gal_weight[in_bin]) / np.sum(gal_weight[in_bin])
        )
    logsm_obs_weighted_mean = np.array(logsm_obs_weighted_mean)

    return logmp_bin_centers, logsm_obs_weighted_mean


def plot_smhm_z(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    data_label,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    in_situ=False,
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 4.5
    fig_height = 4
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), constrained_layout=True)

    labelsize = 10
    fontsize = 14
    # alpha = 0.25
    # colors_z = ["#001219", "#0a7a80", "#80cca8", "#c8b44a", "#c87820", "#9b1d20"]
    colors_z = [
        "#4B2D8F",  # 19-3748 Deep Violet
        "#2055A4",  # 19-4150 Classic Blue
        "#009473",  # 17-5335 Arcadia
        "#D4A017",  # 14-0951 Saffron
        "#E8601C",  # 16-1358 Flame
        "#9B1B30",  # 19-1757 Chili Pepper
    ]

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))

        """fit"""
        lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
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
        if in_situ:
            logsm_obs = phot_kern_results.logsm_obs_in_situ
            (
                logmp_bin_centers_fit,
                logsm_obs_weighted_mean_fit,
            ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
        else:
            logsm_obs = phot_kern_results.logsm_obs
            p_merge = phot_kern_results.p_merge
            (
                logmp_bin_centers_fit,
                logsm_obs_weighted_mean_fit,
            ) = get_logsm_obs_weighted_mean(
                lc_data.logmp_obs, logsm_obs, gal_weight * (1 - p_merge)
            )

        ax.plot(
            logmp_bin_centers_fit,
            logsm_obs_weighted_mean_fit,
            label=z_min_label + " < z < " + z_max_label,
            color=colors_z[zbin],
        )

    ax.set_xlim(11, 14)
    ax.set_ylim(8, 12)

    ax.set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

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

    if in_situ:
        ax.set_ylabel(
            r"<log$_{10}$ (M$_{*, in-situ}$ [M$_{\odot}$])>", fontsize=fontsize
        )
    else:
        ax.set_ylabel(r"<log$_{10}$ (M$_{*}$ [M$_{\odot}$])>", fontsize=fontsize)
    ax.legend(fontsize=10, loc="lower right")

    if in_situ:
        fig.savefig(
            savedir + "/" + data_label + "_smhm_z_insitu.png",
            dpi=300,
        )
    else:
        fig.savefig(
            savedir + "/" + data_label + "_smhm_z.png",
            dpi=300,
        )

    if plt_show:
        plt.show()
    plt.close()


def plot_smhm(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    data_label,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    in_situ=False,
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 1.42 * n_z_bins
    fig_height = 2
    fig, ax = plt.subplots(
        1, len(zbins), figsize=(fig_width, fig_height), constrained_layout=True
    )

    labelsize = 10
    fontsize = 10
    labelsize = 10
    # alpha = 0.25

    for zbin in range(n_z_bins):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))
        ax[zbin].set_title(z_min_label + " < z < " + z_max_label)

        """default"""
        lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
            ran_key,
            DEFAULT_PARAM_COLLECTION,
            z_min,
            z_max,
            num_halos,
            ssp_data,
            tcurves,
            mag_thresh=mag_thresh,
            frac_cat=frac_cat,
        )
        if in_situ:
            logsm_obs = phot_kern_results.logsm_obs_in_situ
            (
                logmp_bin_centers_default,
                logsm_obs_weighted_mean_default,
            ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
        else:
            logsm_obs = phot_kern_results.logsm_obs
            p_merge = phot_kern_results.p_merge
            (
                logmp_bin_centers_default,
                logsm_obs_weighted_mean_default,
            ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
            # * (1 - p_merge)
            # )

        ax[zbin].plot(
            logmp_bin_centers_default,
            logsm_obs_weighted_mean_default,
            label="default",
            color="#FFB689",
            lw=2,
        )

        """fit"""
        lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
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
        if in_situ:
            logsm_obs = phot_kern_results.logsm_obs_in_situ
            (
                logmp_bin_centers_fit,
                logsm_obs_weighted_mean_fit,
            ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
        else:
            logsm_obs = phot_kern_results.logsm_obs
            p_merge = phot_kern_results.p_merge

            (
                logmp_bin_centers_fit,
                logsm_obs_weighted_mean_fit,
            ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
            # * (1 - p_merge)
            # )

            # (
            #     _,
            #     logsm_obs_weighted_mean_default_wout_pmerge_fit,
            # ) = get_logsm_obs_weighted_mean(lc_data.logmp_obs, logsm_obs, gal_weight)
            # ax[zbin].plot(
            #     logmp_bin_centers_default,
            #     logsm_obs_weighted_mean_default_wout_pmerge_fit,
            #     label="wout p_merge weight",
            #     color="#D6353D",
            #     lw=2,
            # )

        ax[zbin].plot(
            logmp_bin_centers_fit,
            logsm_obs_weighted_mean_fit,
            label="fit",
            color="#61C0BF",
            lw=2,
        )

        ax[zbin].set_xlim(11, lc_data.logmp_obs.max())
        ax[zbin].set_ylim(8, 13)
        ax[zbin].set_xticks([11, 12, 13, 14, 15])
        ax[zbin].set_yticks([8, 9, 10, 11, 12])

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

        ax[zbin].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

    if in_situ:
        ax[0].set_ylabel(
            r"<log$_{10}$ (M$_{*, in-situ}$ [M$_{\odot}$])>", fontsize=fontsize
        )
    else:
        ax[0].set_ylabel(r"<log$_{10}$ (M$_{*}$ [M$_{\odot}$])>", fontsize=fontsize)
    ax[-1].legend(fontsize=7, loc="lower right")

    if in_situ:
        fig.savefig(
            savedir + "/" + data_label + "_smhm_insitu.png",
            dpi=300,
        )
    else:
        fig.savefig(
            savedir + "/" + data_label + "_smhm.png",
            dpi=300,
        )

    if plt_show:
        plt.show()
    plt.close()
