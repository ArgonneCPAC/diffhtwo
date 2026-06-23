import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def _mean_occupation(
    logmp_obs,
    halo_indx,
    is_central,
    cen_weight,
    sat_weight,
    logsm_obs,
    dbin=0.25,
    logsm_thresh=10,
):
    logmhost = logmp_obs[halo_indx]
    logmhost_bins = np.arange(logmhost.min(), logmhost.max() + dbin, dbin)
    logmhost_bin_centers = (logmhost_bins[:-1] + logmhost_bins[1:]) / 2

    tot_occ = []
    cen_occ = []
    for b in range(len(logmhost_bins) - 1):
        in_bin = (logmhost > logmhost_bins[b]) & (logmhost <= logmhost_bins[b + 1])
        in_bin = np.ones(in_bin.size) * in_bin

        nhost = np.sum(in_bin * cen_weight * is_central)

        in_sm_thresh = np.ones(logsm_obs.size) * (logsm_obs > logsm_thresh)

        ntot = np.sum(in_sm_thresh * in_bin * cen_weight * sat_weight)
        ncen = np.sum(in_sm_thresh * in_bin * cen_weight * is_central)

        tot_occ.append(ntot / nhost)
        cen_occ.append(ncen / nhost)

    tot_occ = np.array(tot_occ)
    cen_occ = np.array(cen_occ)

    return logmhost_bin_centers, tot_occ, cen_occ


def plot_hod_sm_thresh(
    ran_key,
    param_collection,
    ssp_data,
    tcurves,
    z_min,
    z_max,
    data_label,
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
    halo_indx = lc_data.halo_indx
    is_central = lc_data.is_central
    cen_weight = lc_data.cen_weight
    sat_weight = lc_data.sat_weight
    logsm_obs = phot_data.logsm_obs

    logsm_thresh = np.array([9, 10, 11])

    fig_width = 5
    fig_height = 4
    # labelsize = 10
    # fontsize = 10
    labelsize = 10
    alpha = 0.75

    colors = ["#0057A8", "#FFCD00", "#DA291C"]

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), constrained_layout=True)
    for i in range(0, len(logsm_thresh)):
        logmp, tot_occ, cen_occ = _mean_occupation(
            logmp_obs,
            halo_indx,
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

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.savefig(
        savedir
        + "/"
        + data_label
        + "_hod_z"
        + z_min_label
        + "-"
        + z_max_label
        + ".png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    plt.close()
