import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def _mean_occupation(
    logmp_obs, is_central, cen_weight, sat_weight, logsm_obs, dbin=0.25, logsm_thresh=10
):
    logmp_bins = np.arange(logmp_obs.min(), logmp_obs.max() + dbin, dbin)
    logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2

    tot_occ = []
    cen_occ = []
    for b in range(len(logmp_bins) - 1):
        in_bin = (logmp_obs > logmp_bins[b]) & (logmp_obs <= logmp_bins[b + 1])
        in_bin = np.ones(in_bin.size) * in_bin

        nhost = np.sum(in_bin * cen_weight * is_central)

        in_sm_thresh = np.ones(logsm_obs.size) * (logsm_obs > logsm_thresh)

        ntot = np.sum(in_sm_thresh * in_bin * cen_weight * sat_weight)
        ncen = np.sum(in_sm_thresh * in_bin * cen_weight * is_central)

        tot_occ.append(ntot / nhost)
        cen_occ.append(ncen / nhost)

    tot_occ = np.array(tot_occ)
    cen_occ = np.array(cen_occ)

    return logmp_bin_centers, tot_occ, cen_occ


def plot_hod_sm_thresh(
    ran_key,
    param_collection,
    ssp_data,
    tcurves,
    z_min,
    z_max,
    savedir,
    num_halos=1000,
    plt_show=True,
):
    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )
    logmp_obs = lc_data.logmp_obs
    is_central = lc_data.is_central
    cen_weight = lc_data.cen_weight
    sat_weight = lc_data.sat_weight
    logsm_obs = phot_data.logsm_obs

    logsm_thresh = np.array([8, 10])

    fig_width = 5
    fig_height = 4
    # labelsize = 10
    # fontsize = 10
    # labelsize = 10
    alpha = 0.75

    # "#0a7a80", "#80cca8", "#c87820",

    colors = ["#001219", "#9b1d20"]

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), constrained_layout=True)
    for i in range(0, len(logsm_thresh)):
        logmp, tot_occ, cen_occ = _mean_occupation(
            logmp_obs,
            is_central,
            cen_weight,
            sat_weight,
            logsm_obs,
            logsm_thresh=logsm_thresh[i],
        )
        ax.plot(
            logmp,
            tot_occ,
            label=r"M$_{*}$ [M$_{\odot}$] > " + str(logsm_thresh[i]),
            color=colors[i],
            alpha=alpha,
        )
        ax.plot(
            logmp,
            cen_occ,
            ls="--",
            color=colors[i],
            alpha=alpha,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"log$_{10}$ (M$_{halo}$ [M$_{\odot}$])")
    ax.set_ylabel(r"N$_{gal}$ | > M$_{*}$")
    ax.set_title(str(z_min) + " < z < " + str(z_max))
    ax.legend(fontsize=8)

    if plt_show:
        plt.show()
    plt.close()
