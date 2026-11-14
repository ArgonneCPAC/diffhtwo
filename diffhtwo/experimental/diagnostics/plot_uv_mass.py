import matplotlib.pyplot as plt
import numpy as np
from diffsky.lc_generators.lc_data_phot import mc_lc_data_phot
from diffsky.sed_models.ou26.lc_phot import mc_lc_photline
from jax import random as jran

from diffhtwo.experimental.kernels.lc_photline_kern import lc_photline_kern

from ..kernels.sfr_tau import get_logsfr_tau
from ..kernels.uv_luminosity import compute_M_UV_from_L_UV
from .plot_phot import minerva_colors
from .plot_utils import make_thresholded_reduce_C_function

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_uv_mstar(
    ran_key,
    param_collection,
    zbins,
    ssp_data_uv,
    tcurves,
    line_wave_table_uv,
    num_halos,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_max=15.0,
    n_z_phot_table=2,
    mc_merge=1,
    xlim=(8, 12),
    ylim=(-17, -22),
    gridsize=(70, 50),
    vmin=-12,
    vmax=-8,
    plt_show=True,
    nrows=2,
    ncols=4,
):
    n_z_bins = len(zbins)
    assert n_z_bins <= nrows * ncols

    phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
        ran_key,
        param_collection,
        zbins[0][0],
        zbins[-1][-1],
        num_halos,
        ssp_data_uv,
        tcurves,
        line_wave_table_uv,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        n_z_phot_table=n_z_phot_table,
        mc_merge=mc_merge,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight
    M_uv = compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 0])

    lgsm_obs = phot_data.logsm_obs

    lgsfr = get_logsfr_tau(phot_data, lc_data, ssp_data_uv)
    lgssfr = lgsfr - lgsm_obs

    unmerged = phot_data.p_merge < 0.5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = axes.ravel()
    labelsize = 9
    fontsize = 10

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        z_sel = (lc_data.z_obs > z_min) & (lc_data.z_obs < z_max)

        M_uv_selected = M_uv[unmerged & z_sel]
        lgsm_obs_selected = lgsm_obs[unmerged & z_sel]
        lgssfr_selected = lgssfr[unmerged & z_sel]
        gal_weight_selected = gal_weight[unmerged & z_sel]

        idx = np.arange(len(lgssfr_selected))

        def reduce_C_function(i):
            i = np.asarray(i, dtype=int)
            return np.log10(
                np.average(10 ** lgssfr_selected[i], weights=gal_weight_selected[i])
            )

        hb = ax[z].hexbin(
            lgsm_obs_selected,
            M_uv_selected,
            C=idx,
            reduce_C_function=reduce_C_function,
            cmap="coolwarm_r",
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
            vmin=vmin,
            vmax=vmax,
        )
        ax[z].set_title(f"z={z_label}", fontsize=labelsize)

        ax[z].set_xlim(xlim)
        ax[z].set_ylim(ylim)

        ax[z].minorticks_on()
        ax[z].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[z].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    for a in ax[n_z_bins:]:
        a.set_visible(False)

    fig.supxlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=fontsize)
    fig.colorbar(
        hb, ax=axes, label=r"log$_{10}$ (sSFR$_{100\,\mathrm{Myr}}$ [yr$^{-1}$])"
    )

    fig.savefig(savedir + "/" + run_label + "_uv_mstar.png", dpi=600)

    if plt_show:
        plt.show()
    plt.close()


def plot_uv_mstar_density(
    ran_key,
    param_collection,
    zbins,
    ssp_data_uv,
    tcurves,
    line_wave_table_uv,
    num_halos,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_max=15.0,
    n_z_phot_table=2,
    mc_merge=1,
    xlim=(8, 12),
    ylim=(-17, -22),
    gridsize=(70, 50),
    threshold=1e-6,
    plt_show=True,
    nrows=2,
    ncols=4,
):
    n_z_bins = len(zbins)
    assert n_z_bins <= nrows * ncols

    phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
        ran_key,
        param_collection,
        zbins[0][0],
        zbins[-1][-1],
        num_halos,
        ssp_data_uv,
        tcurves,
        line_wave_table_uv,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        n_z_phot_table=n_z_phot_table,
        mc_merge=mc_merge,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight
    M_uv = compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 0])
    lgsm_obs = phot_data.logsm_obs
    unmerged = phot_data.p_merge < 0.5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = axes.ravel()
    labelsize = 9
    fontsize = 10

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        sel = unmerged & (lc_data.z_obs > z_min) & (lc_data.z_obs < z_max)

        hb = ax[z].hexbin(
            lgsm_obs[sel],
            M_uv[sel],
            C=gal_weight[sel],
            reduce_C_function=make_thresholded_reduce_C_function(
                gal_weight[sel], threshold
            ),
            bins="log",
            cmap="viridis",
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
        )
        ax[z].set_title(f"z={z_label}", fontsize=labelsize)

        ax[z].set_xlim(xlim)
        ax[z].set_ylim(ylim)

        ax[z].minorticks_on()
        ax[z].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[z].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    for a in ax[n_z_bins:]:
        a.set_visible(False)

    fig.supxlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=fontsize)
    fig.colorbar(hb, ax=axes, label=r"f")

    fig.savefig(savedir + "/" + run_label + "_uv_mstar_density.png", dpi=600)

    if plt_show:
        plt.show()
    plt.close()


def plot_uv_mhalo_density(
    ran_key,
    param_collection,
    zbins,
    ssp_data_uv,
    tcurves,
    line_wave_table_uv,
    num_halos,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_max=15.0,
    n_z_phot_table=2,
    mc_merge=1,
    xlim=(10, 13),
    ylim=(-16, -22),
    gridsize=(50, 30),
    threshold=1e-6,
    plt_show=True,
    nrows=2,
    ncols=4,
):
    n_z_bins = len(zbins)
    assert n_z_bins <= nrows * ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = axes.ravel()
    labelsize = 9
    fontsize = 10

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data_uv,
            tcurves,
            line_wave_table_uv,
            lgmp_min=lgmp_min,
            lgmp_max=lgmp_max,
            n_z_phot_table=n_z_phot_table,
            mc_merge=mc_merge,
        )
        lgmp_obs = lc_data.logmp_obs
        gal_weight = lc_data.cen_weight * lc_data.sat_weight
        M_uv = compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 0])
        unmerged = phot_data.p_merge < 0.5

        hb = ax[z].hexbin(
            lgmp_obs[unmerged],
            M_uv[unmerged],
            C=gal_weight[unmerged],
            reduce_C_function=make_thresholded_reduce_C_function(
                gal_weight[unmerged], threshold
            ),
            bins="log",
            cmap="viridis",
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
        )
        ax[z].set_title(f"z={z_label}", fontsize=labelsize)

        ax[z].set_xlim(xlim)
        ax[z].set_ylim(ylim)

        ax[z].minorticks_on()
        ax[z].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[z].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    for a in ax[n_z_bins:]:
        a.set_visible(False)

    fig.supxlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=fontsize)
    fig.colorbar(hb, ax=axes, label=r"f")

    fig.savefig(savedir + "/" + run_label + "_uv_mhalo_density.png", dpi=600)

    if plt_show:
        plt.show()
    plt.close()


def plot_uv_mhalo_density_mc(
    ran_key,
    param_collection,
    tcurves,
    ssp_data_uv,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_sub_min=10.5,
    lgmp_max=15.0,
    area_scale_down=1,
    n_z_phot_table=2,
    mc_merge=1,
    xlim=(10, 13),
    ylim=(-16, -22),
    plt_show=True,
    nrows=2,
    ncols=4,
):
    zbins = [
        (0.3, 0.5),
        (0.7, 0.9),
        (1.3, 1.7),
        (2.0, 2.5),
        (2.5, 3.5),
        (3.5, 4.5),
        (4.5, 5.5),
        (5.5, 6.5),
    ]
    n_z_bins = len(zbins)
    assert n_z_bins <= nrows * ncols

    sky_areas_degsq = np.array([1.0, 0.40, 0.12, 0.087, 0.045, 0.048, 0.054, 0.060])
    sky_areas_degsq = sky_areas_degsq * area_scale_down

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = axes.ravel()
    labelsize = 9
    fontsize = 10

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        args = (
            lc_halo_key,
            z_min,
            z_max,
            lgmp_min,
            lgmp_sub_min,
            sky_areas_degsq[z],
            ssp_data_uv,
            tcurves,
            z_phot_table,
        )
        lc_data = mc_lc_data_phot(*args, lgmp_max=lgmp_max)

        ran_key, sed_key = jran.split(ran_key, 2)
        photline_data, phot_randoms = mc_lc_photline(
            sed_key, lc_data, mc_merge, param_collection=param_collection
        )

        lgmp_obs = lc_data.logmp_obs
        lgsm_obs = photline_data.logsm_obs

        M_uv = compute_M_UV_from_L_UV(photline_data.UV)

        lgsfr100 = get_logsfr_tau(photline_data, lc_data, ssp_data_uv)
        lgssfr100 = lgsfr100 - lgsm_obs

        unmerged = photline_data.p_merge < 0.5

        sc = ax[z].scatter(
            lgmp_obs[unmerged],
            M_uv[unmerged],
            s=1,
            c=lgssfr100[unmerged],
            vmin=-11,
            vmax=-8,
            cmap="coolwarm_r",
        )
        ax[z].set_title(f"z={z_label}", fontsize=labelsize)

        ax[z].set_xlim(xlim)
        ax[z].set_ylim(ylim)

        ax[z].minorticks_on()
        ax[z].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[z].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    for a in ax[n_z_bins:]:
        a.set_visible(False)

    fig.colorbar(
        sc, ax=axes, extend="both", label=r"log$_{10}$ (sSFR$_{100}$ [yr$^{-1}$])"
    )

    fig.supxlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=fontsize)

    fig.savefig(savedir + "/" + run_label + "_uv_mhalo_density_mc.png", dpi=600)

    if plt_show:
        plt.show()
    plt.close()


def plot_uv_scatter_mhalo(
    ran_key,
    param_collection,
    zbins,
    ssp_data_uv,
    tcurves,
    line_wave_table_uv,
    num_halos,
    run_label,
    savedir,
    lgmp_min=10.0,
    lgmp_max=11.0,
    n_z_phot_table=2,
    mc_merge=1,
    plt_show=True,
    nrows=2,
    ncols=4,
):
    n_z_bins = len(zbins)
    assert n_z_bins <= nrows * ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 3.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = axes.ravel()
    labelsize = 9
    fontsize = 10
    legendsize = 7

    m_uv_bin_edges = np.arange(-24, -1, 0.5)
    m_uv_bin_centers = 0.5 * (m_uv_bin_edges[1:] + m_uv_bin_edges[:-1])

    lgmp_bin_edges = np.arange(lgmp_min, lgmp_max, 0.5)

    n_bins = len(lgmp_bin_edges) - 1
    colors = minerva_colors(n_bins)

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
            ran_key,
            param_collection,
            z_min,
            z_max,
            num_halos,
            ssp_data_uv,
            tcurves,
            line_wave_table_uv,
            lgmp_min=lgmp_min,
            lgmp_max=lgmp_max,
            n_z_phot_table=n_z_phot_table,
            mc_merge=mc_merge,
        )
        gal_weight = lc_data.cen_weight * lc_data.sat_weight
        M_uv = compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 0])
        unmerged = phot_data.p_merge < 0.5

        for bin in range(n_bins):
            lgmp_bin_min = lgmp_bin_edges[bin]
            lgmp_bin_max = lgmp_bin_edges[bin + 1]

            in_bin = (lc_data.logmp_obs > lgmp_bin_min) & (
                lc_data.logmp_obs < lgmp_bin_max
            )

            w = gal_weight[unmerged & in_bin]
            muv = M_uv[unmerged & in_bin]
            mean = np.average(muv, weights=w)
            sigma = np.sqrt(np.average((muv - mean) ** 2, weights=w))

            pdf, _ = np.histogram(
                muv,
                weights=w,
                bins=m_uv_bin_edges,
                density=True,
            )

            lgmp_label = np.median((lgmp_bin_min, lgmp_bin_max))
            ax[z].plot(
                m_uv_bin_centers,
                pdf,
                color=colors[bin],
                lw=1.5,
                label=rf"$\sigma_{{M_\mathrm{{UV}}}}$(M$_h$={lgmp_label:.2f}): {sigma:.2f}",
            )
            ax[z].fill_between(m_uv_bin_centers, pdf, color=colors[bin], alpha=0.2)

        ax[z].set_title(f"z={z_label}", fontsize=labelsize)
        ax[z].minorticks_on()
        ax[z].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[z].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        ax[z].legend(fontsize=legendsize, frameon=False, loc="upper right")

    for a in ax[n_z_bins:]:
        a.set_visible(False)

    fig.supxlabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=fontsize)
    fig.supylabel("PDF", fontsize=fontsize)

    fig.savefig(
        f"{savedir}/{run_label}_uv_scatter_mhalo_{lgmp_min}-{lgmp_max}.png", dpi=600
    )

    if plt_show:
        plt.show()
    plt.close()
