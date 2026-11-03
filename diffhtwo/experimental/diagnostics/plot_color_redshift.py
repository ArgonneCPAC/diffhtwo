import matplotlib.pyplot as plt
import numpy as np
from diffsky.lc_generators.lc_data_phot import mc_lc_data_phot
from diffsky.sed_models.ou26.lc_phot import mc_lc_phot
from jax import random as jran
from matplotlib import colors
from matplotlib.patches import Rectangle

from ..kernels.sfr_tau import get_logsfr_tau
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


def _box(fig, ax, c0, c1, label, color, labelsize=12):
    p0 = ax[-1][c0].get_position()
    p1 = ax[0][c1].get_position()
    fig.add_artist(
        Rectangle(
            (p0.x0, p0.y0),
            p1.x1 - p0.x0,
            p1.y1 - p0.y0,
            transform=fig.transFigure,
            fill=False,
            edgecolor=color,
            lw=2,
        )
    )
    fig.text(
        (p0.x0 + p1.x1) / 2,
        p1.y1 - 0.01,
        label,
        ha="center",
        va="top",
        fontsize=labelsize,
        fontweight="bold",
        color=color,
        bbox=dict(fc="w", ec="none", pad=1),
    )


def plot_color_z(
    ran_key,
    param_collection,
    z_min,
    z_max,
    data,
    ssp_data,
    run_label,
    savedir,
    parent_idx=-1,
    sky_area_degsq=0.1,
    lgmp_min=11.0,
    lgmp_sub_min=11.0,
    mc_merge=1,
    gridsize=(200, 20),
    n_z_phot_table=25,
    plt_show=True,
):
    ran_key, lc_halo_key = jran.split(ran_key, 2)
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    args = (
        lc_halo_key,
        z_min,
        z_max,
        lgmp_min,
        lgmp_sub_min,
        sky_area_degsq,
        ssp_data,
        data.filter_info.tcurves,
        z_phot_table,
    )

    lc_data = mc_lc_data_phot(*args)
    ran_key, sed_key = jran.split(ran_key, 2)
    phot_info, __, __ = mc_lc_phot(
        sed_key, lc_data, mc_merge, param_collection=param_collection
    )

    z_obs = lc_data.z_obs
    logmp_obs = lc_data.logmp_obs
    logsm_obs = phot_info.logsm_obs
    logsfr_100Myr = get_logsfr_tau(phot_info, lc_data, ssp_data)
    logssfr_100Myr = logsfr_100Myr - logsm_obs

    fig_width = 7.1
    fig_height = 5.4
    labelsize = 12
    n_gal, n_color = data.color.shape
    fig, ax = plt.subplots(
        n_color,
        5,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0},
    )
    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.075, left=0.1, right=0.99, top=1)

    hbd_list, hb0_list, hb1_list, hb2_list, hb3_list = [], [], [], [], []
    for c in range(n_color):
        col_idx1, col_idx2 = data.color_idx[c]
        model_sel1 = (
            phot_info.obs_mags[:, col_idx1] > data.filter_info.mag_thresh[col_idx1][0]
        ) & (phot_info.obs_mags[:, col_idx1] < data.filter_info.mag_thresh[col_idx1][1])
        model_sel2 = (
            phot_info.obs_mags[:, col_idx2] > data.filter_info.mag_thresh[col_idx2][0]
        ) & (phot_info.obs_mags[:, col_idx2] < data.filter_info.mag_thresh[col_idx2][1])

        model_sel_parent = (
            phot_info.obs_mags[:, parent_idx]
            > data.filter_info.mag_thresh[parent_idx][0]
        ) & (
            phot_info.obs_mags[:, parent_idx]
            < data.filter_info.mag_thresh[parent_idx][1]
        )

        model_sel = (
            model_sel1 & model_sel2 & model_sel_parent & (phot_info.p_merge < 0.5)
        )

        model_color = phot_info.obs_mags[:, col_idx1] - phot_info.obs_mags[:, col_idx2]

        data_sel1 = (
            data.mags[:, col_idx1] > data.filter_info.mag_thresh[col_idx1][0]
        ) & (data.mags[:, col_idx1] < data.filter_info.mag_thresh[col_idx1][1])
        data_sel2 = (
            data.mags[:, col_idx2] > data.filter_info.mag_thresh[col_idx2][0]
        ) & (data.mags[:, col_idx2] < data.filter_info.mag_thresh[col_idx2][1])
        data_sel_parent = (
            data.mags[:, parent_idx] > data.filter_info.mag_thresh[parent_idx][0]
        ) & (data.mags[:, parent_idx] < data.filter_info.mag_thresh[parent_idx][1])
        data_sel = data_sel1 & data_sel2 & data_sel_parent

        data_color = data.mags[:, col_idx1] - data.mags[:, col_idx2]

        combined = np.concatenate([model_color, data_color])
        y_min, y_max = np.percentile(combined, [1, 99])

        hbd = ax[c][0].hexbin(
            data.redshift[data_sel],
            data_color[data_sel],
            C=np.full(data_sel.sum(), 1 / data.data_sky_area_degsq),
            reduce_C_function=np.sum,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            norm=colors.LogNorm(),
            edgecolors="none",
            rasterized=True,
        )
        hb0 = ax[c][1].hexbin(
            z_obs[model_sel],
            model_color[model_sel],
            C=np.full(model_sel.sum(), 1 / sky_area_degsq),
            reduce_C_function=np.sum,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            norm=colors.LogNorm(),
            edgecolors="none",
            rasterized=True,
        )
        hb1 = ax[c][2].hexbin(
            z_obs[model_sel],
            model_color[model_sel],
            C=logsm_obs[model_sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="YlGnBu",
            mincnt=1,
            vmin=9,
            vmax=11.5,
            edgecolors="none",
            rasterized=True,
        )
        hb2 = ax[c][3].hexbin(
            z_obs[model_sel],
            model_color[model_sel],
            C=logmp_obs[model_sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="YlGnBu",
            mincnt=1,
            vmin=11,
            vmax=12.5,
            edgecolors="none",
            rasterized=True,
        )
        hb3 = ax[c][4].hexbin(
            z_obs[model_sel],
            model_color[model_sel],
            C=logssfr_100Myr[model_sel],
            reduce_C_function=np.median,
            gridsize=gridsize,
            cmap="coolwarm_r",
            mincnt=1,
            vmin=-11,
            vmax=-9,
            edgecolors="none",
            rasterized=True,
        )
        hbd_list.append(hbd)
        hb0_list.append(hb0)
        hb1_list.append(hb1)
        hb2_list.append(hb2)
        hb3_list.append(hb3)

        ax[c][0].set_ylabel(data.color_label[c], fontsize=labelsize)

        for i in range(0, 5):
            ax[c][i].minorticks_on()
            ax[c][i].tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1,
                labelsize=labelsize,
            )
            ax[c][i].tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                labelsize=labelsize,
            )
            ax[c][i].set_xlim(z_min - 0.05, z_max + 0.05)
            ax[c][i].set_ylim(y_min, y_max)
            if i != 0:
                ax[c][i].tick_params(labelleft=False)
            if c != n_color - 1:
                ax[c][i].tick_params(labelbottom=False)

    counts = np.concatenate([hb.get_array() for hb in hbd_list + hb0_list])
    norm = colors.LogNorm(vmin=counts.min(), vmax=counts.max())
    for hb in hbd_list + hb0_list:
        hb.set_norm(norm)

    fig.colorbar(
        hbd_list[0],
        ax=ax[:, 0],
        location="top",
        label=r"n$_{FENIKS}$ [deg$^{-2}$]",
        shrink=0.85,
        pad=0.01,
    )
    fig.colorbar(
        hb0_list[0],
        ax=ax[:, 1],
        location="top",
        label=r"n$_{Diffsky}$ [deg$^{-2}$]",
        shrink=0.85,
        pad=0.01,
    )
    fig.colorbar(
        hb1_list[0],
        ax=ax[:, 2],
        location="top",
        label="median\n" + r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])",
        shrink=0.85,
        pad=0.01,
    )
    fig.colorbar(
        hb2_list[0],
        ax=ax[:, 3],
        location="top",
        label="median\n" + r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])",
        shrink=0.85,
        pad=0.01,
    )
    fig.colorbar(
        hb3_list[0],
        ax=ax[:, 4],
        location="top",
        label="median\n" + r"log$_{10}$ (sSFR$_{100Myr}$ [yr$^{-1}$])",
        shrink=0.85,
        pad=0.01,
    )

    fig.supxlabel("redshift", fontsize=labelsize)

    _box(fig, ax, 0, 0, "FENIKS", "tab:blue", labelsize)
    _box(fig, ax, 1, 4, "Diffsky", "tab:orange", labelsize)

    fig.savefig(savedir + "/" + run_label + "_color_redshift.png", dpi=600)
    if plt_show:
        plt.show()
    plt.close()
