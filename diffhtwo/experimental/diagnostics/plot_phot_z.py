import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_colors_z(
    ran_key,
    param_collection,
    dataset,
    data_label,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    savedir,
    plt_show=True,
):
    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        dataset.filter_info.tcurves,
    )

    mag_thresh = np.array(dataset.filter_info.mag_thresh)
    mag_mask = phot_data.obs_mags[:, 0] < mag_thresh[0]
    for b in range(1, len(mag_thresh)):
        mag_mask *= phot_data.obs_mags[:, b] < mag_thresh[b]

    obs_mags = phot_data.obs_mags[mag_mask]
    z_obs = lc_data.z_obs[mag_mask]

    _, n_bands = phot_data.obs_mags.shape
    n_colors = n_bands - 1
    fig, ax = plt.subplots(n_colors, 1, figsize=(5, 12.0), constrained_layout=True)
    c_raw = "k"
    c_lh = "deepskyblue"

    for i in range(n_colors):
        ax[i].scatter(
            dataset.dataset[:, -1],
            dataset.dataset[:, i],
            s=0.5,
            alpha=0.2,
            label=data_label,
            c=c_raw,
            rasterized=True,
        )

        ax[i].set_ylabel(dataset.dataset_dim_labels[i], fontsize=14)
        if i != n_colors - 1:
            ax[i].set_xticks([])

        # ylim = (
        #     np.percentile(dataset.dataset[:, i], 2),
        #     np.percentile(dataset.dataset[:, i], 98),
        # )

        # ax[i].set_ylim(ylim)
        ax[i].set_xlim(z_min, z_max)

        color = obs_mags[:, i] - obs_mags[:, i + 1]
        ax[i].scatter(
            z_obs, color, s=1, alpha=0.5, c=c_lh, label="diffsky", rasterized=True
        )

        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=10,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=10,
        )

    ax[-1].set_xlabel("redshift", fontsize=14)
    ax[0].legend(loc="best", fontsize=14)
    fig.savefig(
        savedir
        + "/"
        + data_label
        + "_colors_z"
        + str(z_min)
        + "-"
        + str(z_max)
        + ".png",
        dpi=300,
    )
    if plt_show:
        plt.show()
