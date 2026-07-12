import matplotlib.pyplot as plt
import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from matplotlib.lines import Line2D

from ..kernels.sfh_rapid_q import get_logsfr_obs

plt.rc("font", family="serif", serif=["Times New Roman"])


def _get_f_q(logsm_obs, logssfr_obs):
    logsm_bins = np.arange(logsm_obs.min(), logsm_obs.max() + 0.25, 0.25)
    logsm_bin_centers = (logsm_bins[:-1] + logsm_bins[1:]) / 2

    quenched = logssfr_obs < -11

    fq_list = []
    for b in range(0, len(logsm_bins) - 1):
        in_bin = (logsm_obs > logsm_bins[b]) & (logsm_obs <= logsm_bins[b + 1])

        if logssfr_obs[in_bin].size > 0:
            f_q = logssfr_obs[quenched & in_bin].size / logssfr_obs[in_bin].size
            fq_list.append(f_q)
        else:
            fq_list.append(0.0)

    f_q_arr = np.array(fq_list)

    return f_q_arr, logsm_bin_centers


def plot_f_q(
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
        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
            gal_weight,
            is_central,
        ) = get_logsfr_obs(
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

        logssfr_obs = logsfr_obs - logsm_obs

        f_q_default, logsm_bin_centers_default = _get_f_q(logsm_obs, logssfr_obs)
        ax[zbin].plot(
            logsm_bin_centers_default,
            f_q_default,
            label="default",
            color="#FFB689",
            lw=2,
        )

        f_q_default_cen, logsm_bin_centers_default_cen = _get_f_q(
            logsm_obs[is_central == 1], logssfr_obs[is_central == 1]
        )
        ax[zbin].plot(
            logsm_bin_centers_default_cen,
            f_q_default_cen,
            color="#FFB689",
            lw=1,
            ls="--",
        )

        f_q_default_sat, logsm_bin_centers_default_sat = _get_f_q(
            logsm_obs[is_central != 1], logssfr_obs[is_central != 1]
        )
        ax[zbin].plot(
            logsm_bin_centers_default_sat,
            f_q_default_sat,
            color="#FFB689",
            lw=1,
            ls=":",
        )

        """fit"""
        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
            gal_weight,
            is_central,
        ) = get_logsfr_obs(
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

        logssfr_obs = logsfr_obs - logsm_obs

        f_q_fit, logsm_bin_centers_fit = _get_f_q(logsm_obs, logssfr_obs)
        ax[zbin].plot(
            logsm_bin_centers_fit,
            f_q_fit,
            label="fit",
            color="#61C0BF",
            lw=2,
        )

        f_q_fit_cen, logsm_bin_centers_fit_cen = _get_f_q(
            logsm_obs[is_central == 1], logssfr_obs[is_central == 1]
        )
        ax[zbin].plot(
            logsm_bin_centers_fit_cen,
            f_q_fit_cen,
            color="#61C0BF",
            lw=1,
            ls="--",
        )

        f_q_fit_sat, logsm_bin_centers_fit_sat = _get_f_q(
            logsm_obs[is_central != 1], logssfr_obs[is_central != 1]
        )
        ax[zbin].plot(
            logsm_bin_centers_fit_sat,
            f_q_fit_sat,
            color="#61C0BF",
            lw=1,
            ls=":",
        )

        ax[zbin].set_xlim(8, 12)
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

    ax[0].set_ylabel(r"f$_{q}$", fontsize=fontsize)
    handles, labels = ax[-1].get_legend_handles_labels()

    cen_handle = Line2D([], [], linestyle="--", color="gray", label="cen")
    handles.append(cen_handle)

    sat_handle = Line2D([], [], linestyle=":", color="gray", label="sat")
    handles.append(sat_handle)

    ax[-1].legend(handles=handles, fontsize=6, loc="upper left", framealpha=0.5)

    fig.savefig(
        savedir + "/" + data_label + "_f_q.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()
