import matplotlib.pyplot as plt
import numpy as np
from diffstar.diffstarpop.kernels import satquenchpop_model as sqpm
from matplotlib import lines as mlines

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

mblue = "tab:blue"
morange = "tab:orange"
mred = "tab:red"

tarr = np.linspace(-10, 15, 40_000)
qprob_cen = 0.35
host_configs = [(12.0, mblue), (13.0, morange), (15.0, mred)]
mu_configs = [(-0.5, "--"), (-3.0, "-")]

p_merge = [0.9, 0.6, 0.3]
log_sm = [8, 9, 10]
logmhost_infall = [12, 13, 14]
colors = [mred, morange, mblue]


def generate_sat_plots(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    num_halos=10000,
    plt_show=True,
):
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

    plot_sat_ssfr_mhost(*args, plt_show=plt_show)
    plot_sat_ssfr_sm(*args, plt_show=plt_show)
    plot_sat_lgfburst_mhost(*args, plt_show=plt_show)
    plot_sat_lgfburst_sm(*args, plt_show=plt_show)
    plot_satquench_model(
        param_collection.diffstarpop_params,
        model_nickname,
        savedir,
        plt_show=plt_show,
    )


def plot_satquench_model(diffstarpop_params, model_nickname, savedir, plt_show=True):
    sqpm_params = sqpm.SatQuenchPopParams(
        diffstarpop_params.qp_lgmh_crit,
        diffstarpop_params.td_lgmhc,
        diffstarpop_params.td_mlo,
        diffstarpop_params.td_mhi,
        diffstarpop_params.qphi_lgmu_crit,
        diffstarpop_params.lgmu_lo_mh_lo,
        diffstarpop_params.lgmu_hi_mh_lo,
        diffstarpop_params.lgmu_lo_mh_hi,
        diffstarpop_params.lgmu_hi_mh_hi,
    )
    param_configs = [
        (sqpm.DEFAULT_SATQUENCHPOP_PARAMS, "default"),
        (sqpm_params, model_nickname),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ylim = (0.001, 1)
    xlim = (-3.9, 5)

    i = 0
    for params, title in param_configs:
        for lgmh, color in host_configs:
            for lgmu, ls in mu_configs:
                q = sqpm.get_qprob_sat(params, lgmu, lgmh, tarr, qprob_cen)
                ax[i].plot(tarr, q, color=color, ls=ls)
        ax[i].plot(
            np.linspace(*xlim, 100),
            np.zeros(100) + qprob_cen,
            ":",
            color="k",
            label=r"${\rm P_{Q, cen}}$",
        )
        ax[i].plot(np.zeros(100), np.linspace(*ylim, 100), ":", color="k")
        ax[i].set_ylim(ylim)
        ax[i].set_xlim(xlim)
        ax[i].set_title(title)

        xlabel = ax[i].set_xlabel(r"${\rm Gyr\ since\ infall}$")
        ylabel = ax[i].set_ylabel(r"$P_{\rm quench}$")

        i += 1

    red_line = mlines.Line2D([], [], ls="-", c=mred, label=r"$m_{\rm host}=15$")
    orange_line = mlines.Line2D([], [], ls="-", c=morange, label=r"$m_{\rm host}=13$")
    blue_line = mlines.Line2D([], [], ls="-", c=mblue, label=r"$m_{\rm host}=12$")
    dashed_line = mlines.Line2D([], [], ls="--", c="gray", label=r"$\mu=1/3$")
    solid_line = mlines.Line2D([], [], ls="-", c="gray", label=r"$\mu=1/1000$")
    black_line = mlines.Line2D([], [], ls=":", c="k", label=r"${\rm P_{Q, cen}}$")
    leg0 = ax[0].legend(handles=[red_line, orange_line, blue_line], loc="lower left")
    ax[0].add_artist(leg0)

    ax[0].legend(handles=[dashed_line, solid_line, black_line], loc="lower right")

    fig.savefig(
        savedir + "/satquench_" + model_nickname + ".png",
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )
    if plt_show:
        plt.show()
    plt.close()


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
    plt_show=True,
):
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
    if plt_show:
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
    plt_show=True,
):
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
    if plt_show:
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
    plt_show=True,
):
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
    if plt_show:
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
    plt_show=True,
):
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
    if plt_show:
        plt.show()
    plt.close()
