import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_massive_cen_colors(
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
    logsm_obs_thresh=11,
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

    sm_cut = (phot_kern_results.logsm_obs > logsm_obs_thresh) & (
        lc_data.is_central == 1
    )

    obs_mags = phot_kern_results.obs_mags
    obs_mags_in_situ = phot_kern_results.obs_mags_in_situ

    n_gals, n_bands = obs_mags.shape

    obs_colors = obs_mags[:, 0 : n_bands - 1] - obs_mags[:, 1:n_bands]
    obs_colors_in_situ = (
        obs_mags_in_situ[:, 0 : n_bands - 1] - obs_mags_in_situ[:, 1:n_bands]
    )

    n_colors = obs_colors.shape[1]
    fig, ax = plt.subplots(1, n_colors, figsize=(14, 3))
    fig.subplots_adjust(wspace=0.3, bottom=0.22, left=0.05, right=0.99, top=0.85)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))

    fig.suptitle(
        "centrals w/ logsm_obs > "
        + str(logsm_obs_thresh)
        + " | "
        + z_min_label
        + " < z < "
        + z_max_label
    )
    for c in range(0, n_colors):
        std = np.std(obs_colors_in_situ[:, c][sm_cut])
        med = np.median(obs_colors_in_situ[:, c][sm_cut])
        bins = np.linspace(
            med - (3 * std),
            med + (3 * std),
            20,
        )
        ax[c].hist(
            obs_colors_in_situ[:, c][sm_cut],
            weights=weights[sm_cut],
            bins=bins,
            density=True,
            alpha=0.5,
            label="in-situ",
        )

        ax[c].hist(
            obs_colors[:, c][sm_cut],
            weights=weights[sm_cut],
            bins=bins,
            density=True,
            alpha=0.5,
            label="in+ex-situ",
        )
        ax[c].set_xlabel(dimension_labels[c])
    ax[0].set_ylabel("PDF")
    plt.legend(fontsize=8)

    fig.savefig(
        savedir
        + "/cen_"
        + model_nickname
        + "_colors_massive_z"
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
