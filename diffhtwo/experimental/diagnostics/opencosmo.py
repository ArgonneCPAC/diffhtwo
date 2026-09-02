import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..lc_utils import zbin_vol


def compare_mocks_mag_z(
    mock1,
    mock2,
    mock_label1,
    mock_label2,
    redshift_column,
    mag_column,
    savedir,
    extent=(0.05, 4.0, 10, 40),
    plt_show=True,
):
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 4), constrained_layout=True)

    ax[0].set_title(mock_label1)
    sel1 = mock1["p_merge"] < 0.9
    ax[0].hexbin(
        mock1[redshift_column][sel1],
        mock1[mag_column][sel1],
        mincnt=3,
        norm="log",
        extent=extent,
        rasterized=True,
    )
    ax[1].set_title(mock_label2)
    sel2 = mock2["p_merge"] < 0.9
    ax[1].hexbin(
        mock2[redshift_column][sel2],
        mock2[mag_column][sel2],
        mincnt=3,
        norm="log",
        extent=extent,
        rasterized=True,
    )
    fig.supylabel(mag_column)
    fig.supxlabel(redshift_column)

    fig.savefig(savedir + "/" + mock_label1 + "_v_" + mock_label2 + ".png", dpi=400)
    if plt_show:
        plt.show()
    plt.close()


def plot_halpha(
    hizels, mock, mock_sky_area_degsq, mock_label, cosmo_params, savedir, plt_show=True
):
    fig, ax = plt.subplots(1, figsize=(4.3, 4), constrained_layout=True)
    colors_z = [
        "#0a7a80",  # teal
        "#80cca8",  # mint
        "#c8b44a",  # warm gold
        "#c87820",  # amber
    ]
    ylim = (-5.0, -1.0)
    xlim = (39.8, 43.5)

    alpha = 0.75
    lw = 2

    lg_Lbin_edges = hizels.lg_Lbin_edges[0]
    lg_phi_data = hizels.lg_phi_data[0]
    z = hizels.z[0]
    dz = hizels.dz[0]

    offsets_z = np.array([0, 0.2, 0.5, 0.8])

    for i in range(0, 4):
        lgL_bin_centers = (lg_Lbin_edges[i][1:] + lg_Lbin_edges[i][:-1]) / 2
        ax.errorbar(
            lgL_bin_centers,
            lg_phi_data[i][0] + offsets_z[i],
            lg_phi_data[i][1],
            color=colors_z[i],
            fmt="s",
            markersize=5,
            alpha=alpha,
        )

        zmin = float(z[i] - (dz[i] / 2))
        zmax = float(z[i] + (dz[i] / 2))

        vol = zbin_vol(mock_sky_area_degsq, zmin, zmax, cosmo_params)

        z_sel = (mock["redshift_obs"] > zmin) & (
            mock["redshift_obs"] < zmax
        )  # &  (mock["p_merge"] < 0.9)

        N, _ = np.histogram(
            np.log10(mock["Ba_alpha_6563"][z_sel]), bins=lg_Lbin_edges[i]
        )
        lg_halpha_LF = np.log10(N / vol)
        ax.plot(
            lgL_bin_centers,
            lg_halpha_LF + offsets_z[i],
            color=colors_z[i],
            alpha=alpha,
            label=" z = " + str(hizels.z[0][i]),
            lw=lw,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1,
        labelsize=10,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
        labelsize=10,
    )

    handles_z, labels_z = ax.get_legend_handles_labels()
    handles = [
        Line2D([], [], color="k", lw=lw, label=mock_label),
        ax.errorbar(
            [],
            [],
            yerr=[[0.2], [0.2]],  # vertical error bar
            fmt="s",
            color="k",
            markersize=6,
            linestyle="none",
            lw=lw,
            label="Sobral+13 (HiZELS)",
        ),
    ]
    handles = handles_z + handles
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=10,
    )

    ax.set_xlabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=14)
    ax.set_ylabel("log$_{10}($\u03d5 [Mpc$^{-3}$])", fontsize=14)
    fig.savefig(savedir + "/" + mock_label + "_halpha.png", dpi=400)
    if plt_show:
        plt.show()
    plt.close()
