import matplotlib.pyplot as plt
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from scipy.stats import binned_statistic

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def get_median_logsm_obs(logmp_obs, logsm_obs):
    logmp_bins = np.arange(11.0, logmp_obs.max() + 0.25, 0.25)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    logsm_16, __, __ = binned_statistic(
        logmp_obs, logsm_obs, bins=logmp_bins, statistic=lambda x: np.percentile(x, 16)
    )
    logsm_50, __, __ = binned_statistic(
        logmp_obs, logsm_obs, bins=logmp_bins, statistic="median"
    )
    logsm_84, __, __ = binned_statistic(
        logmp_obs, logsm_obs, bins=logmp_bins, statistic=lambda x: np.percentile(x, 84)
    )
    return logmp_bin_centers, logsm_16, logsm_50, logsm_84


def plot_smhm(
    ran_key,
    param_collection,
    zbins,
    num_halos,
    ssp_data,
    tcurves,
    mag_thresh,
    frac_cat,
    data_label,
    savedir,
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
    alpha = 0.25

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
        (
            logmp_bin_centers_default,
            logsm_16_default,
            logsm_50_default,
            logsm_84_default,
        ) = get_median_logsm_obs(lc_data.logmp_obs, phot_kern_results.logsm_obs)

        ax[zbin].plot(
            logmp_bin_centers_default,
            logsm_50_default,
            label="default",
            color="#FFB689",
        )
        ax[zbin].fill_between(
            logmp_bin_centers_default,
            logsm_16_default,
            logsm_84_default,
            alpha=alpha,
            color="#FFB689",
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

        (
            logmp_bin_centers_fit,
            logsm_16_fit,
            logsm_50_fit,
            logsm_84_fit,
        ) = get_median_logsm_obs(lc_data.logmp_obs, phot_kern_results.logsm_obs)

        ax[zbin].plot(logmp_bin_centers_fit, logsm_50_fit, label="fit", color="#61C0BF")
        ax[zbin].fill_between(
            logmp_bin_centers_fit,
            logsm_16_fit,
            logsm_84_fit,
            alpha=alpha,
            color="#61C0BF",
        )

        ax[zbin].set_xlabel("logmp_obs", fontsize=fontsize)
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

    ax[0].set_ylabel("logsm_obs", fontsize=fontsize)
    ax[-1].legend(fontsize=7, loc="lower right")

    fig.savefig(
        savedir + "/" + data_label + "_smhm.png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    plt.close()
