import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_merging_sat_colors(
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
    num_halos=10000,
    logsm_obs_thresh=6,
    p_merge_thresh=0.5,
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

    pmerge_thresh_lo = 0.1
    pmerge_thresh_hi = 0.5
    sel_pmerge_lo = (
        (phot_kern_results.logsm_obs_in_situ > logsm_obs_thresh)
        & (lc_data.is_central != 1)
        & (phot_kern_results.p_merge < pmerge_thresh_lo)
    )

    sel_pmerge_hi = (
        (phot_kern_results.logsm_obs > logsm_obs_thresh)
        & (lc_data.is_central != 1)
        & (phot_kern_results.p_merge > pmerge_thresh_hi)
    )

    obs_mags_in_situ = phot_kern_results.obs_mags_in_situ

    n_gals, n_bands = obs_mags_in_situ.shape

    obs_colors_in_situ = (
        obs_mags_in_situ[:, 0 : n_bands - 1] - obs_mags_in_situ[:, 1:n_bands]
    )

    n_colors = obs_colors_in_situ.shape[1]
    fig, ax = plt.subplots(1, n_colors, figsize=(14, 3))
    fig.subplots_adjust(wspace=0.3, bottom=0.22, left=0.05, right=0.99, top=0.85)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.suptitle(
        "sats w/ logsm_obs_in_situ > "
        + str(logsm_obs_thresh)
        + " | "
        + z_min_label
        + " < z < "
        + z_max_label
    )
    for c in range(0, n_colors):
        std = np.std(obs_colors_in_situ[:, c][sel_pmerge_lo])
        med = np.median(obs_colors_in_situ[:, c][sel_pmerge_lo])
        bins = np.linspace(
            med - (3 * std),
            med + (3 * std),
            20,
        )
        ax[c].hist(
            obs_colors_in_situ[:, c][sel_pmerge_lo],
            weights=weights[sel_pmerge_lo],
            bins=bins,
            density=True,
            alpha=0.5,
            label="p_merge < " + str(pmerge_thresh_lo),
            color="tab:blue",
        )

        ax[c].hist(
            obs_colors_in_situ[:, c][sel_pmerge_hi],
            weights=weights[sel_pmerge_hi],
            bins=bins,
            density=True,
            alpha=0.5,
            label="p_merge > " + str(pmerge_thresh_hi),
            color="tab:orange",
        )
        ax[c].set_xlabel(dimension_labels[c])
    ax[0].set_ylabel("PDF")
    plt.legend(fontsize=8)

    fig.savefig(
        savedir
        + "/merging_sat_"
        + model_nickname
        + "_insitu_colors_z"
        + z_min_label
        + "-"
        + z_max_label
        + ".png"
    )
    if plt_show:
        plt.show()
    plt.close()
