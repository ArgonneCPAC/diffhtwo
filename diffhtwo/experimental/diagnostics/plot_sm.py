import matplotlib.pyplot as plt
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_insitu_sm_obs(
    ran_key,
    param_collection,
    z_min,
    z_max,
    dimension_labels,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    num_halos=1000,
    plt_show=True,
):
    fig, ax = plt.subplots(1, figsize=(5, 5))

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.suptitle(z_min_label + " < z < " + z_max_label)

    """fit"""
    lc_data, phot_kern_results, weights = multiband_lc_phot_kern(
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

    bins = np.linspace(
        phot_kern_results.logsm_obs_in_situ.min(),
        phot_kern_results.logsm_obs_in_situ.max(),
        50,
    )

    cen = lc_data.is_central == 1
    sat = lc_data.is_central != 1

    ax.hist(
        phot_kern_results.logsm_obs_in_situ[sat],
        bins=bins,
        label="fit sat",
        color="tab:orange",
        histtype="step",
    )
    ax.hist(
        phot_kern_results.logsm_obs_in_situ[cen],
        bins=bins,
        label="fit cen",
        color="tab:blue",
        histtype="step",
    )

    """default"""
    lc_data, phot_kern_results, weights = multiband_lc_phot_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )

    cen = lc_data.is_central == 1
    sat = lc_data.is_central != 1

    ax.hist(
        phot_kern_results.logsm_obs_in_situ[sat],
        bins=bins,
        label="default sat",
        color="tab:orange",
        histtype="step",
        ls="--",
    )
    ax.hist(
        phot_kern_results.logsm_obs_in_situ[cen],
        bins=bins,
        label="default cen",
        color="tab:blue",
        histtype="step",
        ls="--",
    )

    ax.set_xlabel("logsm_obs_in_situ")
    ax.set_ylabel("#")
    ax.set_xlim(0, 14)
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(
        savedir
        + "/insitu_sm_obs_"
        + model_nickname
        + "_z"
        + z_min_label
        + "-"
        + z_max_label
        + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    if plt_show:
        plt.show()
    plt.close()


def plot_sm_obs(
    ran_key,
    param_collection,
    z_min,
    z_max,
    dimension_labels,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    num_halos=1000,
    plt_show=True,
):
    fig, ax = plt.subplots(1, figsize=(5, 5))

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.suptitle(z_min_label + " < z < " + z_max_label)

    """fit"""
    lc_data, phot_kern_results, weights = multiband_lc_phot_kern(
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

    bins = np.linspace(
        phot_kern_results.logsm_obs.min(),
        phot_kern_results.logsm_obs.max(),
        50,
    )

    cen = lc_data.is_central == 1
    sat = lc_data.is_central != 1

    ax.hist(
        phot_kern_results.logsm_obs[sat],
        bins=bins,
        label="fit sat",
        color="tab:orange",
        histtype="step",
    )
    ax.hist(
        phot_kern_results.logsm_obs[cen],
        bins=bins,
        label="fit cen",
        color="tab:blue",
        histtype="step",
    )

    """default"""
    lc_data, phot_kern_results, weights = multiband_lc_phot_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )

    cen = lc_data.is_central == 1
    sat = lc_data.is_central != 1

    ax.hist(
        phot_kern_results.logsm_obs[sat],
        bins=bins,
        label="default sat",
        color="tab:orange",
        histtype="step",
        ls="--",
    )
    ax.hist(
        phot_kern_results.logsm_obs[cen],
        bins=bins,
        label="default cen",
        color="tab:blue",
        histtype="step",
        ls="--",
    )

    ax.set_xlabel("logsm_obs")
    ax.set_ylabel("#")
    ax.set_xlim(0, 14)
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(
        savedir
        + "/sm_obs_"
        + model_nickname
        + "_z"
        + z_min_label
        + "-"
        + z_max_label
        + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    if plt_show:
        plt.show()
    plt.close()
