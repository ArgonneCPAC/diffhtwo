import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

mblue = "tab:blue"
morange = "tab:orange"
mred = "tab:red"


def generate_sat_plots(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    num_halos=10000,
):
    lc_data, phot_kern_results, weights = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )
    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    args = (
        ran_key,
        lc_data,
        phot_kern_results,
        weights,
        param_collection,
        z_min_label,
        z_max_label,
        model_nickname,
        savedir,
    )

    plot_sat_ssfr_mhost(*args)
    plot_sat_ssfr_sm(*args)
    plot_sat_lgfburst_mhost(*args)
    plot_sat_lgfburst_sm(*args)


def plot_sat_ssfr_mhost(
    ran_key,
    lc_data,
    phot_kern_results,
    weights,
    param_collection,
    z_min_label,
    z_max_label,
    model_nickname,
    savedir,
):
    p_merge = [0.9, 0.6, 0.3]
    logmhost_infall = [12, 13, 14]
    colors = [mred, morange, mblue]
    fig, ax = plt.subplots(3, len(p_merge), figsize=(14, 14))
    fig.suptitle(
        "Satellite ssfr | " + z_min_label + " < z < " + z_max_label, y=0.91, fontsize=20
    )
    ssfr_bins = np.arange(-13, -9, 0.2)

    xlim = (-14, -9)
    for m in range(0, len(logmhost_infall)):
        for p in range(0, len(p_merge)):
            sat = lc_data.is_central != 1
            merging_sat = (
                (sat)
                & (phot_kern_results.p_merge > p_merge[p])
                & (lc_data.logmhost_infall > logmhost_infall[m])
            )
            ax[m][p].hist(
                phot_kern_results.logssfr_obs[sat],
                weights=weights[sat],
                bins=ssfr_bins,
                alpha=0.8,
                histtype="step",
                color=colors[p],
            )
            ax[m][p].hist(
                phot_kern_results.logssfr_obs[merging_sat],
                weights=weights[merging_sat],
                bins=ssfr_bins,
                alpha=0.5,
                color=colors[p],
                label="logmhost_infall > "
                + str(logmhost_infall[m])
                + "\np_merge > "
                + str(p_merge[p]),
            )
            ax[m][p].set_xlim(xlim)
            ax[m][p].legend()
            ax[m][p].set_yscale("log")

    ax[1][0].set_ylabel("#", fontsize=18)
    ax[2][1].set_xlabel("logssfr", fontsize=18)
    fig.savefig(
        savedir + "/sat_" + model_nickname + "_ssfr_mhost.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
    plt.close()


def plot_sat_ssfr_sm(
    ran_key,
    lc_data,
    phot_kern_results,
    weights,
    param_collection,
    z_min_label,
    z_max_label,
    model_nickname,
    savedir,
):
    p_merge = [0.9, 0.6, 0.3]
    log_sm = [8, 9, 10]
    colors = [mred, morange, mblue]

    fig, ax = plt.subplots(3, len(p_merge), figsize=(14, 14))
    fig.suptitle(
        "Satellite ssfr | " + z_min_label + " < z < " + z_max_label, y=0.91, fontsize=20
    )
    ssfr_bins = np.arange(-13, -9, 0.2)

    xlim = (-14, -9)
    for m in range(0, len(log_sm)):
        for p in range(0, len(p_merge)):
            sat = lc_data.is_central != 1
            merging_sat = (
                (sat)
                & (phot_kern_results.p_merge > p_merge[p])
                & (phot_kern_results.logsm_obs > log_sm[m])
            )
            ax[m][p].hist(
                phot_kern_results.logssfr_obs[sat],
                weights=weights[sat],
                bins=ssfr_bins,
                alpha=0.8,
                histtype="step",
                color=colors[p],
            )
            ax[m][p].hist(
                phot_kern_results.logssfr_obs[merging_sat],
                weights=weights[merging_sat],
                bins=ssfr_bins,
                alpha=0.5,
                color=colors[p],
                label="logsm > " + str(log_sm[m]) + "\np_merge > " + str(p_merge[p]),
            )
            ax[m][p].set_xlim(xlim)
            ax[m][p].legend()
            ax[m][p].set_yscale("log")

    ax[1][0].set_ylabel("#", fontsize=18)
    ax[2][1].set_xlabel("logssfr", fontsize=18)
    fig.savefig(
        savedir + "/sat_" + model_nickname + "_ssfr_sm.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
    plt.close()


def plot_sat_lgfburst_mhost(
    ran_key,
    lc_data,
    phot_kern_results,
    weights,
    param_collection,
    z_min_label,
    z_max_label,
    model_nickname,
    savedir,
):
    p_merge = [0.9, 0.6, 0.3]
    logmhost_infall = [10, 11, 12]
    colors = [mred, morange, mblue]

    fig, ax = plt.subplots(3, len(p_merge), figsize=(14, 14))
    fig.suptitle(
        "Satellite lgfburst | " + z_min_label + " < z < " + z_max_label,
        y=0.91,
        fontsize=20,
    )
    for m in range(0, len(logmhost_infall)):
        for p in range(0, len(p_merge)):
            sat = lc_data.is_central != 1
            merging_sat = (
                (sat)
                & (phot_kern_results.p_merge > p_merge[p])
                & (phot_kern_results.logsm_obs > logmhost_infall[m])
            )
            ax[m][p].hist(
                phot_kern_results.lgfburst[sat],
                weights=weights[sat],
                bins=20,
                alpha=0.5,
                color=colors[p],
                histtype="step",
            )
            ax[m][p].hist(
                phot_kern_results.lgfburst[merging_sat],
                weights=weights[merging_sat],
                bins=20,
                alpha=0.5,
                color=colors[p],
                label="logmhost_infall > "
                + str(logmhost_infall[m])
                + "\np_merge > "
                + str(p_merge[p]),
            )
            ax[m][p].set_yscale("log")
            ax[m][p].legend()
    ax[1][0].set_ylabel("#", fontsize=18)
    ax[2][1].set_xlabel("lgfburst", fontsize=18)
    fig.savefig(
        savedir + "/sat_" + model_nickname + "_lgfburst_mhost.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
    plt.close()


def plot_sat_lgfburst_sm(
    ran_key,
    lc_data,
    phot_kern_results,
    weights,
    param_collection,
    z_min_label,
    z_max_label,
    model_nickname,
    savedir,
):
    p_merge = [0.9, 0.6, 0.3]
    log_sm = [6, 7, 8]
    colors = [mred, morange, mblue]

    fig, ax = plt.subplots(3, len(p_merge), figsize=(14, 14))
    fig.suptitle(
        "Satellite lgfburst | " + z_min_label + " < z < " + z_max_label,
        y=0.91,
        fontsize=20,
    )
    for m in range(0, len(log_sm)):
        for p in range(0, len(p_merge)):
            sat = lc_data.is_central != 1
            merging_sat = (
                (sat)
                & (phot_kern_results.p_merge > p_merge[p])
                & (phot_kern_results.logsm_obs > log_sm[m])
            )
            ax[m][p].hist(
                phot_kern_results.lgfburst[sat],
                weights=weights[sat],
                bins=20,
                alpha=0.5,
                color=colors[p],
                histtype="step",
            )
            ax[m][p].hist(
                phot_kern_results.lgfburst[merging_sat],
                weights=weights[merging_sat],
                bins=20,
                alpha=0.5,
                color=colors[p],
                label="logsm > " + str(log_sm[m]) + "\np_merge > " + str(p_merge[p]),
            )
            ax[m][p].set_yscale("log")
            ax[m][p].legend()
    ax[1][0].set_ylabel("#", fontsize=18)
    ax[2][1].set_xlabel("lgfburst", fontsize=18)
    fig.savefig(
        savedir + "/sat_" + model_nickname + "_lgfburst_sm.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
    plt.close()
