import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..kernels.sfh_rapid_q import get_logsfr_obs
from ..tab_blue_orange_cmap import make_cmap
from ..utils import weighted_median
from .plot_utils import make_thresholded_reduce_C_function, percentile_norm

cmap = make_cmap()

plt.rc("font", family="serif", serif=["Times New Roman"])

LOGMP_OBS_MIN, LOGMP_OBS_MAX = 10.5, 14.5


pantone_colors = [
    "#2D3142",  # deep indigo (Pantone Graphite-ish)
    "#4F5D75",  # blue fog
    "#BB5A81",  # radiant orchid
    "#EF8354",  # coral
    "#F4D06F",  # buttercup gold
]

pantone_cmap = LinearSegmentedColormap.from_list("pantone_dusk", pantone_colors, N=256)

calm_colors = [
    "#1B3A4B",  # deep slate blue
    "#3E6680",  # muted steel blue
    "#7DA6A0",  # sage teal
    "#B8CFC1",  # soft sage
    "#EAE3D2",  # warm sand
]

calm_cmap = LinearSegmentedColormap.from_list("calm_sage", calm_colors, N=256)


def _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight):
    logsm_bins = np.arange(logsm_obs.min(), logsm_obs.max() + 0.25, 0.25)
    logsm_bin_centers = (logsm_bins[:-1] + logsm_bins[1:]) / 2

    logsfr_obs_weighted_mean = []
    for b in range(0, len(logsm_bins) - 1):
        in_bin = (logsm_obs > logsm_bins[b]) & (logsm_obs <= logsm_bins[b + 1])
        logsfr_obs_weighted_mean.append(
            np.nansum(logsfr_obs[in_bin] * gal_weight[in_bin])
            / np.nansum(gal_weight[in_bin])
        )
    logsfr_obs_weighted_mean = np.array(logsfr_obs_weighted_mean)

    return logsm_bin_centers, logsfr_obs_weighted_mean


def _get_logsfr_obs_weighted_median(logm_obs, logsfr_obs, gal_weight, t_obs, t_q):
    sf = t_q > t_obs

    logm_bins = np.arange(logm_obs.min(), logm_obs.max() + 0.25, 0.25)
    logm_bin_centers = (logm_bins[:-1] + logm_bins[1:]) / 2

    logsfr_obs_weighted_median = []
    for b in range(0, len(logm_bins) - 1):
        in_bin = (logm_obs > logm_bins[b]) & (logm_obs <= logm_bins[b + 1])
        sel = in_bin & sf

        if sel.sum() > 0:
            logsfr_obs_weighted_median.append(
                weighted_median(logsfr_obs[sel], gal_weight[sel])
            )
        else:
            logsfr_obs_weighted_median.append(np.nan)

    logsfr_obs_weighted_median = np.array(logsfr_obs_weighted_median)

    return logm_bin_centers, logsfr_obs_weighted_median


# def plot_sfms(
#     ran_key,
#     param_collection,
#     zbins,
#     num_halos,
#     ssp_data,
#     tcurves,
#     data_label,
#     savedir,
#     mag_thresh=None,
#     frac_cat=None,
#     plt_show=True,
# ):
#     n_z_bins = len(zbins)
#     fig_width = 1.42 * n_z_bins
#     fig_height = 2
#     fig, ax = plt.subplots(
#         1, len(zbins), figsize=(fig_width, fig_height), constrained_layout=True
#     )

#     labelsize = 10
#     fontsize = 10
#     labelsize = 10
#     # alpha = 0.25

#     for zbin in range(n_z_bins):
#         z_min = zbins[zbin][0]
#         z_max = zbins[zbin][1]
#         z_min_label = str(np.round(z_min, 2))
#         z_max_label = str(np.round(z_max, 2))
#         ax[zbin].set_title(z_min_label + " < z < " + z_max_label)

#         """default"""
#         (
#             logsfr_obs,
#             logsm_obs,
#             logsfr_obs_in_situ,
#             logsm_obs_in_situ,
#             gal_weight,
#             is_central,
#             _,
#         ) = get_logsfr_obs(
#             ran_key,
#             DEFAULT_PARAM_COLLECTION,
#             z_min,
#             z_max,
#             num_halos,
#             ssp_data,
#             tcurves,
#             mag_thresh=mag_thresh,
#             frac_cat=frac_cat,
#         )

#         (
#             logsm_bin_centers_default,
#             logsfr_obs_weighted_mean_default,
#         ) = _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight)

#         ax[zbin].plot(
#             logsm_bin_centers_default,
#             logsfr_obs_weighted_mean_default,
#             label="default",
#             color="#FFB689",
#             lw=2,
#         )

#         (
#             logsm_bin_centers_in_situ_default,
#             logsfr_obs_weighted_mean_in_situ_default,
#         ) = _get_logsfr_obs_weighted_mean(
#             logsm_obs_in_situ, logsfr_obs_in_situ, gal_weight
#         )

#         ax[zbin].plot(
#             logsm_bin_centers_in_situ_default,
#             logsfr_obs_weighted_mean_in_situ_default,
#             color="#FFB689",
#             lw=1,
#             ls="--",
#         )

#         """fit"""
#         (
#             logsfr_obs,
#             logsm_obs,
#             logsfr_obs_in_situ,
#             logsm_obs_in_situ,
#             gal_weight,
#             is_central,
#             _,
#         ) = get_logsfr_obs(
#             ran_key,
#             param_collection,
#             z_min,
#             z_max,
#             num_halos,
#             ssp_data,
#             tcurves,
#             mag_thresh=mag_thresh,
#             frac_cat=frac_cat,
#         )

#         (
#             logsm_bin_centers_fit,
#             logsfr_obs_weighted_mean_fit,
#         ) = _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight)

#         ax[zbin].plot(
#             logsm_bin_centers_fit,
#             logsfr_obs_weighted_mean_fit,
#             label="fit",
#             color="#61C0BF",
#             lw=2,
#         )

#         (
#             logsm_bin_centers_in_situ_fit,
#             logsfr_obs_weighted_mean_in_situ_fit,
#         ) = _get_logsfr_obs_weighted_mean(
#             logsm_obs_in_situ, logsfr_obs_in_situ, gal_weight
#         )

#         ax[zbin].plot(
#             logsm_bin_centers_in_situ_fit,
#             logsfr_obs_weighted_mean_in_situ_fit,
#             color="#61C0BF",
#             lw=1,
#             ls="--",
#         )

#         ax[zbin].set_xlim(8, 12)
#         ax[zbin].set_ylim(-3, 2)
#         # ax[zbin].set_xticks([11, 12, 13, 14, 15])
#         # ax[zbin].set_yticks([8, 9, 10, 11, 12])

#         ax[zbin].minorticks_on()
#         ax[zbin].tick_params(
#             which="major",
#             direction="in",
#             top=True,
#             right=True,
#             length=6,
#             width=1,
#             labelsize=labelsize,
#         )
#         ax[zbin].tick_params(
#             which="minor",
#             direction="in",
#             top=True,
#             right=True,
#             length=3,
#             width=0.8,
#             labelsize=labelsize,
#         )

#         ax[zbin].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)

#     ax[0].set_ylabel(r"<log$_{10}$ (SFR [M$_{\odot}$ yr$^{-1}$])>", fontsize=fontsize)
#     dashed_handle = Line2D([], [], linestyle="--", color="gray", label="in-situ only")
#     handles, labels = ax[-1].get_legend_handles_labels()
#     handles.append(dashed_handle)
#     ax[-1].legend(handles=handles, fontsize=7, loc="lower right")

#     fig.savefig(
#         savedir + "/" + data_label + "_sfr_mass.png",
#         dpi=300,
#     )

#     if plt_show:
#         plt.show()
#     plt.close()


def plot_sfms_hexbin(
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
    xlim=(8, 12),
    ylim=(-3, 2),
    xlim_halo=(10, 13.2),
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 7.1
    fig_height = 4.1
    fig, ax = plt.subplots(
        2,
        n_z_bins,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    labelsize = 10
    fontsize = 10
    labelsize = 9

    logsm_arr = np.arange(8, 12, 0.1)
    gridsize = (70, 60)

    # --- pass 1: bin each panel (invisibly) to get real per-bin densities,
    # cached separately per row so each row gets its own optimized norm,
    # and cache the obs arrays so get_logsfr_obs isn't called twice ---
    tmp_fig, tmp_ax = plt.subplots()
    all_c_sm = []
    all_c_halo = []
    cached = []
    for zbin in range(n_z_bins):
        z_min, z_max = zbins[zbin][0], zbins[zbin][1]
        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
            t_q,
            lc_data,
            phot_data,
            gal_weight,
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
        logmp_obs = lc_data.logmp_obs
        t_obs = lc_data.t_obs

        reduce_C_function = make_thresholded_reduce_C_function(gal_weight)

        hb_tmp_sm = tmp_ax.hexbin(
            logsm_obs,
            logsfr_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        )
        hb_tmp_halo = tmp_ax.hexbin(
            logmp_obs,
            logsfr_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            gridsize=gridsize,
            extent=(xlim_halo[0], xlim_halo[1], ylim[0], ylim[1]),
        )

        all_c_sm.append(hb_tmp_sm.get_array())
        all_c_halo.append(hb_tmp_halo.get_array())
        cached.append(
            (
                logsm_obs,
                logsfr_obs,
                gal_weight,
                logmp_obs,
                t_obs,
                t_q,
                reduce_C_function,
            )
        )
    plt.close(tmp_fig)

    norm_sm = percentile_norm(all_c_sm)
    norm_halo = percentile_norm(all_c_halo)

    # --- pass 2: real plotting, shared (but row-specific) norm across panels ---
    for zbin in range(n_z_bins):
        z_med = np.median(zbins[zbin])
        logsfms_leja22 = get_leja22_sfms_at_z(z_med, logsm_arr)

        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_min_label = str(np.round(z_min, 2))
        z_max_label = str(np.round(z_max, 2))
        ax[0, zbin].set_title(z_min_label + " < z < " + z_max_label)

        (
            logsm_obs,
            logsfr_obs,
            gal_weight,
            logmp_obs,
            t_obs,
            t_q,
            reduce_C_function,
        ) = cached[zbin]

        # -- row 0: SFR vs stellar mass --
        hb_sm = ax[0, zbin].hexbin(
            logsm_obs,
            logsfr_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            norm=norm_sm,
            cmap=cmap,
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        )

        # SF main-sequence
        logsm_bin_centers, logsfr_obs_weighted_median = _get_logsfr_obs_weighted_median(
            logsm_obs, logsfr_obs, gal_weight, t_obs, t_q
        )
        ax[0, zbin].scatter(
            logsm_bin_centers,
            logsfr_obs_weighted_median,
            linewidths=1.0,
            s=10,
            facecolors="none",
            edgecolors="k",
            label="median star-forming",
        )
        ax[0, zbin].plot(
            logsm_arr,
            logsfms_leja22,
            c="black",
            alpha=0.8,
            ls="--",
            lw=1.5,
            label="Leja+22 SFMS",
        )
        ax[0, zbin].plot(
            logsm_arr,
            logsfms_leja22 - 1,
            c="#FF073A",
            alpha=0.8,
            ls="--",
            lw=1.5,
            label="1 dex below",
        )
        ax[0, zbin].set_xlim(xlim)
        ax[0, zbin].set_ylim(ylim)
        ax[0, zbin].set_xticks([8, 9, 10, 11, 12])
        ax[0, zbin].set_yticks([-2, -1, 0, 1, 2])
        ax[0, zbin].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)

        # -- row 1: SFR vs halo mass --
        hb_halo = ax[1, zbin].hexbin(
            logmp_obs,
            logsfr_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            norm=norm_halo,
            cmap=cmap,
            gridsize=gridsize,
            extent=(xlim_halo[0], xlim_halo[1], ylim[0], ylim[1]),
        )
        logmp_bin_centers, logsfr_obs_weighted_median = _get_logsfr_obs_weighted_median(
            logmp_obs, logsfr_obs, gal_weight, t_obs, t_q
        )
        ax[1, zbin].scatter(
            logmp_bin_centers,
            logsfr_obs_weighted_median,
            linewidths=1.0,
            s=10,
            facecolors="none",
            edgecolors="k",
        )

        ax[1, zbin].set_ylim(ylim)
        ax[1, zbin].set_xticks(np.arange(xlim_halo[0], xlim_halo[1] + 1, 1))
        ax[1, zbin].set_xlim(xlim_halo)
        ax[1, zbin].set_yticks([-2, -1, 0, 1, 2])
        ax[1, zbin].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)

        for row in range(2):
            ax[row, zbin].minorticks_on()
            ax[row, zbin].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[row, zbin].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )

    handles, labels = ax[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
        fontsize=labelsize,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax[0, 0].set_ylabel(r"log$_{10}$ (SFR [M$_{\odot}$ yr$^{-1}$])", fontsize=fontsize)
    ax[1, 0].set_ylabel(r"log$_{10}$ (SFR [M$_{\odot}$ yr$^{-1}$])", fontsize=fontsize)

    fig.colorbar(hb_sm, ax=ax[0, -1])
    fig.colorbar(hb_halo, ax=ax[1, -1])

    fig.savefig(
        savedir + "/" + data_label + "_sfr_mass_hexbin.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()


a = np.array([0.03746, 0.3448, -0.1156])
b = np.array([0.9605, 0.04990, -0.05984])
c = np.array([0.2516, 1.118, -0.2006])
logMt = np.array([10.22, 0.3826, -0.04491])


def get_param_at_z(z, X):
    return X[0] + (X[1] * z) + (X[2] * (z**2))


def get_leja22_sfms_at_z(z, logM, a=a, b=b, c=c, logMt=logMt):
    a_z = get_param_at_z(z, a)
    b_z = get_param_at_z(z, b)
    c_z = get_param_at_z(z, c)
    logMt_z = get_param_at_z(z, logMt)

    logsfr_above_Mt = a_z * (logM - logMt_z) + c_z
    logsfr_below_Mt = b_z * (logM - logMt_z) + c_z

    logsfr_below_Mt[logM > logMt_z] = logsfr_above_Mt[logM > logMt_z]
    logsfr = logsfr_below_Mt

    return logsfr
