import matplotlib.pyplot as plt
import numpy as np
from diffsky.burstpop import freqburst_mono
from diffsky.experimental.mc_lightcone_generators import mc_lc_photdata
from diffsky.experimental.mc_phot import mc_lc_phot
from jax import random as jran
from matplotlib.ticker import MaxNLocator

from ..kernels.sfr_tau import get_logsfr_100Myr
from ..tab_blue_orange_cmap import make_cmap

cmap = make_cmap()

plt.rc("font", family="serif", serif=["Times New Roman"])
plt.rc(
    "mathtext",
    fontset="custom",
    rm="Times New Roman",
    it="Times New Roman:italic",
    bf="Times New Roman:bold",
)


def _get_mc_is(param_collection, phot_randoms, logsm_obs, logssfr_obs):
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        logsm_obs,
        logssfr_obs,
    )

    mc_is_q = phot_randoms.mc_is_q
    mc_is_ms = ~mc_is_q

    mc_is_burst = phot_randoms.uran_pburst < p_burst
    mc_is_burst = (mc_is_ms) & (mc_is_burst)
    mc_is_ms = (mc_is_ms) & (~mc_is_burst)

    return mc_is_q, mc_is_ms, mc_is_burst


def compare_models_in_mag_z(
    ran_key,
    param_collection1,
    param_collection2,
    z_min,
    z_max,
    dataset,
    ssp_data,
    run_label1,
    run_label2,
    savedir,
    sky_area_degsq=0.1,
    lgmp_min=11,
    lgmp_sub_min=11,
    mc_merge=1,
    gridsize=100,
    plt_show=True,
):
    ran_key, lc_halo_key = jran.split(ran_key, 2)
    z_phot_table = np.linspace(z_min, z_max, 25)
    tcurves = dataset.filter_info.tcurves
    mag_thresh = dataset.filter_info.mag_thresh
    mags_labels = dataset.mags_labels
    z_data = dataset.mags[:, -1]
    args = (
        lc_halo_key,
        z_min,
        z_max,
        lgmp_min,
        lgmp_sub_min,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = mc_lc_photdata(*args)

    z_obs = lc_data.z_obs
    ran_key, sed_key = jran.split(ran_key, 2)

    phot_info1, phot_randoms1, merging_randoms1 = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection1
    )
    phot_info2, phot_randoms2, merging_randoms2 = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection2
    )

    logsfr_100Myr1 = get_logsfr_100Myr(phot_info1, lc_data, ssp_data)
    logsm_obs1 = phot_info1.logsm_obs
    logssfr_obs1 = logsfr_100Myr1 - logsm_obs1
    mc_is_q1, mc_is_ms1, mc_is_burst1 = _get_mc_is(
        param_collection1, phot_randoms1, logsm_obs1, logssfr_obs1
    )

    logsfr_100Myr2 = get_logsfr_100Myr(phot_info2, lc_data, ssp_data)
    logsm_obs2 = phot_info2.logsm_obs
    logssfr_obs2 = logsfr_100Myr2 - logsm_obs2
    mc_is_q2, mc_is_ms2, mc_is_burst2 = _get_mc_is(
        param_collection1, phot_randoms1, logsm_obs2, logssfr_obs2
    )

    p_merge1 = phot_info1.p_merge
    p_merge2 = phot_info2.p_merge
    sel1 = np.ones(len(phot_info1.obs_mags), dtype=bool)
    sel2 = np.ones(len(phot_info2.obs_mags), dtype=bool)

    # for band in range(len(mag_thresh)):
    #     thresh = mag_thresh[band]
    #     sel &= (phot_info.obs_mags[:, band] > thresh[0]) & (
    #         phot_info.obs_mags[:, band] < thresh[1]
    #     )

    sel1 &= p_merge1 < 0.9
    sel2 &= p_merge2 < 0.9
    fig_width = 7.1
    fig_height = 6.6
    n_bands = len(tcurves)
    n_columns = 2
    fig, ax = plt.subplots(
        n_bands,
        n_columns,
        figsize=(fig_width, fig_height),
    )
    fig.subplots_adjust(
        wspace=0, hspace=0, bottom=0.075, left=0.075, right=0.99, top=0.925
    )
    labelsize = 12
    ax[0][0].set_title("SDSS+FENIKS+HiZELS (" + run_label1 + ")")
    ax[0][1].set_title("SDSS+FENIKS (" + run_label2 + ")")
    for f in range(0, n_bands):
        mag_diffsky1 = phot_info1.obs_mags[:, f]
        mag_diffsky2 = phot_info2.obs_mags[:, f]
        # combined = np.concatenate([mag_diffsky, mag_data])
        # y_min, y_max = np.percentile(combined, [1, 99])
        y_min, y_max = 15, 30

        # mc_is_burst
        ax[f][0].scatter(
            z_obs[sel1 & mc_is_burst1],
            mag_diffsky1[sel1 & mc_is_burst1],
            s=1,
            alpha=0.5,
            c="tab:orange",
            label="mc_is_bursty",
            rasterized=True,
        )
        ax[f][1].scatter(
            z_obs[sel2 & mc_is_burst2],
            mag_diffsky2[sel2 & mc_is_burst2],
            s=1,
            alpha=0.5,
            c="tab:orange",
            rasterized=True,
        )

        # mc_is_ms
        ax[f][0].scatter(
            z_obs[sel1 & mc_is_ms1],
            mag_diffsky1[sel1 & mc_is_ms1],
            s=0.005,
            alpha=0.5,
            c="deepskyblue",
            label="mc_is_ms",
            rasterized=True,
        )
        ax[f][1].scatter(
            z_obs[sel2 & mc_is_ms2],
            mag_diffsky2[sel2 & mc_is_ms2],
            s=0.005,
            alpha=0.5,
            c="deepskyblue",
            rasterized=True,
        )

        # mc_is_q
        ax[f][0].scatter(
            z_obs[sel1 & mc_is_q1],
            mag_diffsky1[sel1 & mc_is_q1],
            s=0.005,
            alpha=0.5,
            c="darkred",
            label="mc_is_q",
            rasterized=True,
        )
        ax[f][1].scatter(
            z_obs[sel2 & mc_is_q2],
            mag_diffsky2[sel2 & mc_is_q2],
            s=0.005,
            alpha=0.5,
            c="darkred",
            rasterized=True,
        )

        ax[f][0].set_ylabel(mags_labels[f], fontsize=labelsize)
        if f == 0:
            y_prune = "lower"
        elif f == n_bands - 1:
            y_prune = "upper"
        else:
            y_prune = "both"
        for i in range(0, n_columns):
            ax[f][i].minorticks_on()
            ax[f][i].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[f][i].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )
            ax[f][i].set_xlim(z_min, z_max)
            ax[f][i].set_ylim(y_min, y_max)
            ax[f][i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune=y_prune))
            x_prune = "upper" if i == 0 else "lower"
            ax[f][i].xaxis.set_major_locator(MaxNLocator(nbins=5, prune=x_prune))
            if i != 0:
                ax[f][i].tick_params(labelleft=False)
            if f != n_bands - 1:
                ax[f][i].tick_params(labelbottom=False)

    handles, labels = ax[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    for handle in leg.legend_handles:
        handle.set_sizes([30])

    fig.supxlabel("redshift", fontsize=labelsize)
    fig.savefig(
        savedir + "/" + run_label1 + "_v_" + run_label2 + "_mag_redshift.png", dpi=300
    )
    if plt_show:
        plt.show()
    plt.close()


def compare_models_in_mag_z_sfr(
    ran_key,
    param_collection1,
    param_collection2,
    z_min,
    z_max,
    dataset,
    ssp_data,
    run_label1,
    run_label2,
    savedir,
    sky_area_degsq=0.1,
    lgmp_min=11,
    lgmp_sub_min=11,
    mc_merge=1,
    gridsize=100,
    vmin=-2,
    vmax=2,
    plt_show=True,
):
    ran_key, lc_halo_key = jran.split(ran_key, 2)
    z_phot_table = np.linspace(z_min, z_max, 25)
    tcurves = dataset.filter_info.tcurves
    mag_thresh = dataset.filter_info.mag_thresh
    mags_labels = dataset.mags_labels

    args = (
        lc_halo_key,
        z_min,
        z_max,
        lgmp_min,
        lgmp_sub_min,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = mc_lc_photdata(*args)

    z_obs = lc_data.z_obs
    ran_key, sed_key = jran.split(ran_key, 2)

    phot_info1, phot_randoms1, merging_randoms1 = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection1
    )
    logsfr_100Myr1 = get_logsfr_100Myr(phot_info1, lc_data, ssp_data)

    phot_info2, phot_randoms2, merging_randoms2 = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection2
    )
    logsfr_100Myr2 = get_logsfr_100Myr(phot_info2, lc_data, ssp_data)

    # vmin = min(logsfr_100Myr1.min(), logsfr_100Myr2.min())
    # vmax = max(logsfr_100Myr1.max(), logsfr_100Myr2.max())

    p_merge1 = phot_info1.p_merge
    p_merge2 = phot_info2.p_merge
    sel1 = np.ones(len(phot_info1.obs_mags), dtype=bool)
    sel2 = np.ones(len(phot_info2.obs_mags), dtype=bool)

    # for band in range(len(mag_thresh)):
    #     thresh = mag_thresh[band]
    #     sel &= (phot_info.obs_mags[:, band] > thresh[0]) & (
    #         phot_info.obs_mags[:, band] < thresh[1]
    #     )

    sel1 &= p_merge1 < 0.9
    sel2 &= p_merge2 < 0.9
    fig_width = 7.1
    fig_height = 6.4
    n_bands = len(tcurves)
    n_columns = 2
    fig, ax = plt.subplots(
        n_bands,
        n_columns,
        figsize=(fig_width, fig_height),
    )
    fig.subplots_adjust(
        wspace=0, hspace=0, bottom=0.075, left=0.075, right=0.99, top=0.95
    )
    labelsize = 12
    ax[0][0].set_title("SDSS+FENIKS+HiZELS (" + run_label1 + ")")
    ax[0][1].set_title("SDSS+FENIKS (" + run_label2 + ")")
    for f in range(0, n_bands):
        mag_diffsky1 = phot_info1.obs_mags[:, f]
        mag_diffsky2 = phot_info2.obs_mags[:, f]
        # combined = np.concatenate([mag_diffsky, mag_data])
        # y_min, y_max = np.percentile(combined, [1, 99])
        y_min, y_max = 15, 30

        # sc = ax[f][0].scatter(
        #     z_obs[sel1],
        #     mag_diffsky1[sel1],
        #     s=1,
        #     alpha=0.5,
        #     c=logsfr_100Myr1[sel1],
        #     cmap="viridis",
        #     vmin=vmin,
        #     vmax=vmax,
        #     rasterized=True,
        # )
        # ax[f][1].scatter(
        #     z_obs[sel2],
        #     mag_diffsky2[sel2],
        #     s=1,
        #     alpha=0.5,
        #     c=logsfr_100Myr2[sel2],
        #     cmap="viridis",
        #     vmin=vmin,
        #     vmax=vmax,
        #     rasterized=True,
        # )
        sc = ax[f][0].hexbin(
            z_obs[sel1],
            mag_diffsky1[sel1],
            C=logsfr_100Myr1[sel1],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            extent=(z_min, z_max, y_min, y_max),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax[f][1].hexbin(
            z_obs[sel2],
            mag_diffsky2[sel2],
            C=logsfr_100Myr2[sel2],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            extent=(z_min, z_max, y_min, y_max),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax[f][0].set_ylabel(mags_labels[f], fontsize=labelsize)
        if f == 0:
            y_prune = "lower"
        elif f == n_bands - 1:
            y_prune = "upper"
        else:
            y_prune = "both"
        for i in range(0, n_columns):
            ax[f][i].minorticks_on()
            ax[f][i].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[f][i].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )
            ax[f][i].set_xlim(z_min, z_max)
            ax[f][i].set_ylim(y_min, y_max)
            ax[f][i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune=y_prune))
            x_prune = "upper" if i == 0 else "lower"
            ax[f][i].xaxis.set_major_locator(MaxNLocator(nbins=5, prune=x_prune))
            if i != 0:
                ax[f][i].tick_params(labelleft=False)
            if f != n_bands - 1:
                ax[f][i].tick_params(labelbottom=False)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.075, 0.015, 0.875])
    fig.colorbar(sc, cax=cbar_ax, label="log$_{10}$ SFR$_{100Myr}$")

    fig.supxlabel("redshift", fontsize=labelsize)
    fig.savefig(
        savedir + "/" + run_label1 + "_v_" + run_label2 + "_mag_redshift_sfr.png",
        dpi=300,
    )
    if plt_show:
        plt.show()
    plt.close()
