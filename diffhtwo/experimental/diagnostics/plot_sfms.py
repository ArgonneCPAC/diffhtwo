import cmocean
import matplotlib.pyplot as plt
import numpy as np
from diffsky.merging.merging_kernels import compute_x_tot_from_x_in_situ
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

plt.rc("font", family="serif", serif=["Times New Roman"])


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


def get_logsfr_obs(
    ran_key,
    param_collection,
    z_min,
    z_max,
    num_halos,
    ssp_data,
    tcurves,
    mag_thresh=None,
    frac_cat=None,
):
    lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
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
    logsm_obs_in_situ = phot_kern_results.logsm_obs_in_situ
    logssfr_obs_in_situ = phot_kern_results.logssfr_obs

    logsfr_obs_in_situ = logssfr_obs_in_situ + logsm_obs_in_situ
    sfr_obs_in_situ = 10**logsfr_obs_in_situ

    p_merge = phot_kern_results.p_merge
    sat_weight = lc_data.sat_weight
    halo_indx = lc_data.halo_indx

    sfr_obs = compute_x_tot_from_x_in_situ(
        sfr_obs_in_situ, p_merge, sat_weight, halo_indx
    )
    logsfr_obs = np.log10(sfr_obs)
    logsm_obs = phot_kern_results.logsm_obs

    return logsfr_obs, logsm_obs, logsfr_obs_in_situ, logsm_obs_in_situ, gal_weight


def plot_sfms(
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

        (
            logsm_bin_centers_default,
            logsfr_obs_weighted_mean_default,
        ) = _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight)

        ax[zbin].plot(
            logsm_bin_centers_default,
            logsfr_obs_weighted_mean_default,
            label="default",
            color="#FFB689",
            lw=2,
        )

        (
            logsm_bin_centers_in_situ_default,
            logsfr_obs_weighted_mean_in_situ_default,
        ) = _get_logsfr_obs_weighted_mean(
            logsm_obs_in_situ, logsfr_obs_in_situ, gal_weight
        )

        ax[zbin].plot(
            logsm_bin_centers_in_situ_default,
            logsfr_obs_weighted_mean_in_situ_default,
            color="#FFB689",
            lw=1,
            ls="--",
        )

        """fit"""
        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
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

        (
            logsm_bin_centers_fit,
            logsfr_obs_weighted_mean_fit,
        ) = _get_logsfr_obs_weighted_mean(logsm_obs, logsfr_obs, gal_weight)

        # ax[zbin].scatter(
        #     logsm_obs,
        #     logsfr_obs,
        #     label="fit",
        #     color="#61C0BF",
        # )
        # ax[zbin].scatter(
        #     logsm_obs_in_situ,
        #     logsfr_obs_in_situ,
        #     label="fit",
        #     color="#61C0BF",
        #     alpha=0.5,
        # )

        ax[zbin].plot(
            logsm_bin_centers_fit,
            logsfr_obs_weighted_mean_fit,
            label="fit",
            color="#61C0BF",
            lw=2,
        )

        (
            logsm_bin_centers_in_situ_fit,
            logsfr_obs_weighted_mean_in_situ_fit,
        ) = _get_logsfr_obs_weighted_mean(
            logsm_obs_in_situ, logsfr_obs_in_situ, gal_weight
        )

        ax[zbin].plot(
            logsm_bin_centers_in_situ_fit,
            logsfr_obs_weighted_mean_in_situ_fit,
            color="#61C0BF",
            lw=1,
            ls="--",
        )

        ax[zbin].set_xlim(8, 12)
        ax[zbin].set_ylim(-3, 2)
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

    ax[0].set_ylabel(r"<log$_{10}$ (SFR [M$_{\odot}$ yr$^{-1}$])>", fontsize=fontsize)
    dashed_handle = Line2D([], [], linestyle="--", color="gray", label="in-situ only")
    handles, labels = ax[-1].get_legend_handles_labels()
    handles.append(dashed_handle)
    ax[-1].legend(handles=handles, fontsize=7, loc="lower right")

    fig.savefig(
        savedir + "/" + data_label + "_sfr_mass.png",
        dpi=300,
    )

    if plt_show:
        plt.show()
    plt.close()


def make_thresholded_reduce_C_function(weights, threshold=1e-5):
    total = np.nansum(weights)

    def _reduce(C_bin):
        frac = np.nansum(C_bin) / total
        return frac if frac >= threshold else np.nan

    return _reduce


sfms_z1 = np.array(
    [
        (8.408450704225352, -1.1502590673575126),
        (9.549295774647888, -0.051813471502590414),
        (10.19718309859155, 0.5492227979274613),
        (11.788732394366196, 0.7772020725388602),
    ]
)

sfms_z2 = np.array(
    [
        (8.44055944055944, -1.005181347150259),
        (8.888111888111887, -0.5699481865284977),
        (10.244755244755245, 0.7772020725388602),
        (11.783216783216783, 1.5647668393782386),
    ]
)

sfms_z3 = np.array(
    [
        (8.3943661971831, -0.9430051813471501),
        (8.971830985915496, -0.321243523316062),
        (10.126760563380282, 0.9015544041450778),
        (10.43661971830986, 1.0673575129533681),
        (11.788732394366198, 1.8963730569948185),
    ]
)

sfms_z4 = np.array(
    [
        (8.375886524822693, -0.5699481865284977),
        (9.595744680851064, 0.5077720207253891),
        (10.5177304964539, 1.2538860103626943),
        (11.01418439716312, 1.585492227979275),
        (11.808510638297872, 2.1036269430051813),
    ]
)


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
    plt_show=True,
):
    n_z_bins = len(zbins)
    fig_width = 1.42 * n_z_bins
    fig_height = 1.8
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

        (
            logsfr_obs,
            logsm_obs,
            logsfr_obs_in_situ,
            logsm_obs_in_situ,
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

        reduce_C_function = make_thresholded_reduce_C_function(gal_weight)

        hb = ax[zbin].hexbin(
            logsm_obs,
            logsfr_obs,
            C=gal_weight,
            reduce_C_function=reduce_C_function,
            norm="log",
            cmap=cmocean.cm.dense,
            gridsize=(50, 50),
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            vmin=1e-5,
            vmax=1e-2,
        )

        ax[zbin].set_xlim(xlim)
        ax[zbin].set_ylim(ylim)
        ax[zbin].set_xticks([8, 9, 10, 11, 12])
        ax[zbin].set_yticks([-2, -1, 0, 1, 2])

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

    a = 0.5
    ax[0].plot(sfms_z1[:, 0], sfms_z1[:, 1], c="k", alpha=a)
    ax[1].plot(sfms_z2[:, 0], sfms_z2[:, 1], c="k", alpha=a)
    ax[2].plot(sfms_z3[:, 0], sfms_z3[:, 1], c="k", alpha=a)
    ax[3].plot(sfms_z4[:, 0], sfms_z4[:, 1], c="k", alpha=a)

    ax[0].set_ylabel(r"log$_{10}$ (SFR [M$_{\odot}$ yr$^{-1}$])", fontsize=fontsize)
    fig.colorbar(hb, ax=ax[-1])

    fig.savefig(
        savedir + "/" + data_label + "_sfr_mass_hexbin.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()
