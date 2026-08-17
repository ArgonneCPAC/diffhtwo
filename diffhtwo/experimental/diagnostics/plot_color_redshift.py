import matplotlib.pyplot as plt
import numpy as np
from diffsky.experimental.mc_lightcone_generators import mc_lc_photdata
from diffsky.experimental.mc_phot import mc_lc_phot
from jax import random as jran
from matplotlib import colors

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


def plot_color_z(
    ran_key,
    param_collection,
    z_min,
    z_max,
    dataset,
    ssp_data,
    run_label,
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
    dim_labels = dataset.dataset_dim_labels
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
    ran_key, sed_key = jran.split(ran_key, 2)
    phot_info, __, __ = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection
    )
    z_obs = lc_data.z_obs
    logmp_obs = lc_data.logmp_obs
    logsm_obs = phot_info.logsm_obs
    logsfr_100Myr = get_logsfr_100Myr(phot_info, lc_data, ssp_data)
    sel = np.ones(len(phot_info.obs_mags), dtype=bool)
    for band in range(len(mag_thresh)):
        thresh = mag_thresh[band]
        sel &= (phot_info.obs_mags[:, band] > thresh[0]) & (
            phot_info.obs_mags[:, band] < thresh[1]
        )
    fig_width = 7.1
    fig_height = 6.4
    n_colors = len(tcurves) - 1
    fig, ax = plt.subplots(
        n_colors,
        5,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0},
    )
    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.075, left=0.1, right=0.99, top=1)
    labelsize = 12
    fontsize = 10
    hbd_list, hb0_list, hb1_list, hb2_list, hb3_list = [], [], [], [], []
    for f in range(0, n_colors):
        ax[f][0].set_ylabel(dim_labels[f], fontsize=labelsize)
        color = phot_info.obs_mags[:, f] - phot_info.obs_mags[:, f + 1]
        data_col = dataset.dataset[:, f]
        combined = np.concatenate([color[sel], data_col])
        y_min, y_max = np.percentile(combined, [1, 99])
        hbd = ax[f][0].hexbin(
            dataset.dataset[:, -1],
            data_col,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            norm=colors.LogNorm(),
            edgecolors="none",
            rasterized=True,
        )
        hb0 = ax[f][1].hexbin(
            z_obs[sel],
            color[sel],
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            norm=colors.LogNorm(),
            edgecolors="none",
            rasterized=True,
        )
        hb1 = ax[f][2].hexbin(
            z_obs[sel],
            color[sel],
            C=logsm_obs[sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="YlGnBu",
            mincnt=1,
            vmin=9,
            vmax=11.5,
            edgecolors="none",
            rasterized=True,
        )
        hb2 = ax[f][3].hexbin(
            z_obs[sel],
            color[sel],
            C=logmp_obs[sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="YlGnBu",
            mincnt=1,
            vmin=11,
            vmax=13,
            edgecolors="none",
            rasterized=True,
        )
        hb3 = ax[f][4].hexbin(
            z_obs[sel],
            color[sel],
            C=logsfr_100Myr[sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="coolwarm_r",
            mincnt=1,
            vmin=-2.0,
            vmax=1,
            edgecolors="none",
            rasterized=True,
        )
        hbd_list.append(hbd)
        hb0_list.append(hb0)
        hb1_list.append(hb1)
        hb2_list.append(hb2)
        hb3_list.append(hb3)
        for i in range(0, 5):
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
            ax[f][i].set_xlim(0, z_max + 0.2)
            ax[f][i].set_ylim(y_min, y_max)
            if i != 0:
                ax[f][i].tick_params(labelleft=False)
            if f != n_colors - 1:
                ax[f][i].tick_params(labelbottom=False)

    countsd = np.concatenate([hb.get_array() for hb in hbd_list])
    normd = colors.LogNorm(vmin=countsd.min(), vmax=countsd.max())
    for hb in hbd_list:
        hb.set_norm(normd)
    counts = np.concatenate([hb.get_array() for hb in hb0_list])
    vmin0, vmax0 = counts.min(), counts.max()
    norm0 = colors.LogNorm(vmin=vmin0, vmax=vmax0)
    for hb in hb0_list:
        hb.set_norm(norm0)
    cbd = fig.colorbar(
        hbd_list[0],
        ax=ax[:, 0],
        location="top",
        label=r"N$_{FENIKS}$",
        shrink=0.85,
        pad=0.01,
    )
    cb0 = fig.colorbar(
        hb0_list[0],
        ax=ax[:, 1],
        location="top",
        label=r"N$_{diffsky}$",
        shrink=0.85,
        pad=0.01,
    )
    cb1 = fig.colorbar(
        hb1_list[0],
        ax=ax[:, 2],
        location="top",
        label="median\n" + r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])",
        shrink=0.85,
        pad=0.01,
    )
    cb2 = fig.colorbar(
        hb2_list[0],
        ax=ax[:, 3],
        location="top",
        label="median\n" + r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])",
        shrink=0.85,
        pad=0.01,
    )
    cb3 = fig.colorbar(
        hb3_list[0],
        ax=ax[:, 4],
        location="top",
        label="median\n" + r"log$_{10}$ (SFR$_{100Myr}$ [M$_{\odot}$yr$^{-1}$])",
        shrink=0.85,
        pad=0.01,
    )

    fig.supxlabel("redshift", fontsize=labelsize)
    fig.savefig(savedir + "/" + run_label + "_color_redshift.png", dpi=600)
    if plt_show:
        plt.show()
    plt.close()
