import matplotlib.pyplot as plt
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight):
    logsm_bins = np.arange(logsm_obs.min(), logsm_obs.max() + 0.25, 0.25)
    logsm_bin_centers = (logsm_bins[:-1] + logsm_bins[1:]) / 2

    logsfr_obs_weighted_mean = []
    for b in range(0, len(logsm_bins) - 1):
        in_bin = (logsm_obs > logsm_bins[b]) & (logsm_obs < logsm_bins[b + 1])
        logsfr_obs_weighted_mean.append(
            np.sum(logsm_obs[in_bin] * gal_weight[in_bin]) / np.sum(gal_weight[in_bin])
        )
    logsfr_obs_weighted_mean = np.array(logsfr_obs_weighted_mean)

    return logsm_bin_centers, logsfr_obs_weighted_mean


def plot_sfms(
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
        logsm_obs_default = phot_kern_results.logsm_obs_in_situ
        logsfr_obs_default = (
            phot_kern_results.logssfr_obs + phot_kern_results.logsm_obs_in_situ
        )
        (
            logsm_bin_centers_default,
            logsfr_obs_weighted_mean_default,
        ) = _get_logsfr_obs_weighted_mean(
            logsm_obs_default, logsfr_obs_default, gal_weight
        )

        ax[zbin].plot(
            logsm_bin_centers_default,
            logsfr_obs_weighted_mean_default,
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
        logsm_obs_fit = phot_kern_results.logsm_obs_in_situ
        logsfr_obs_fit = (
            phot_kern_results.logssfr_obs + phot_kern_results.logsm_obs_in_situ
        )
        (
            logsm_bin_centers_fit,
            logsfr_obs_weighted_mean_fit,
        ) = _get_logsfr_obs_weighted_mean(logsm_obs_fit, logsfr_obs_fit, gal_weight)

        ax[zbin].plot(
            logsm_bin_centers_fit,
            logsfr_obs_weighted_mean_fit,
            label="fit",
            color="#61C0BF",
            lw=2,
        )

        # ax[zbin].set_xlim(8, logsm_obs_fit.max())
        # ax[zbin].set_ylim(-3, 2)
        # ax[zbin].set_xticks([11, 12, 13, 14, 15])
        # ax[zbin].set_yticks([8, 9, 10, 11, 12])

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

        ax[zbin].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)

    ax[0].set_ylabel(r"<log$_{10}$ (SFR [M$_{\odot}$ / yr])>", fontsize=fontsize)
    ax[-1].legend(fontsize=7, loc="lower right")

    fig.savefig(
        savedir + "/" + data_label + "_sfr_mass.png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    plt.close()
