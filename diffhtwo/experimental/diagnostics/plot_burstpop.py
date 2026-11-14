"""
Based on diffsky.burstpop.diagnostics.plot_fburstpop
"""
import os

import numpy as np
from diffsky.burstpop import freqburst_mono
from diffsky.burstpop.fburstpop_mono import get_fburst_from_fburstpop_params
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter

from ..kernels.lc_phot_kern import multiband_lc_phot_kern
from ..kernels.lc_photline_kern import lc_photline_kern
from ..kernels.sfr_tau import get_logsfr_tau
from ..kernels.uv_luminosity import compute_M_UV_from_L_UV

DEFAULT_FBURSTPOP_PARAMS = (
    DEFAULT_PARAM_COLLECTION.spspop_params.burstpop_params.fburstpop_params
)
try:
    from matplotlib import pyplot as plt

    plt.rc("font", family="serif", serif=["Times New Roman"])

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"


def make_fburstpop_comparison_plot(
    params,
    params2=DEFAULT_FBURSTPOP_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of the model for Fburst

    Parameters
    ----------
    params : namedtuple
        Instance of fburstpop.FburstPopParams

    params2 : namedtuple, optional
        Instance of fburstpop.FburstPopParams
        Default is set by DEFAULT_FBURSTPOP_PARAMS

    fname : string, optional
        filename of the output figure

    """
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    nsm, nsfr = 250, 250
    logsm_grid = np.linspace(7, 12, nsm)
    logssfr_grid = np.linspace(-13, -8, nsfr)

    X, Y = np.meshgrid(logsm_grid, logssfr_grid)

    Z = np.log10(get_fburst_from_fburstpop_params(params, X, Y))
    Z2 = np.log10(get_fburst_from_fburstpop_params(params2, X, Y))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    (ax0, ax1) = axes

    pcm0 = ax0.pcolor(X, Y, Z, cmap="coolwarm_r", vmin=-4.5, vmax=-2.1)
    fig.colorbar(pcm0, ax=ax0)

    pcm1 = ax1.pcolor(X, Y, Z2, cmap="coolwarm_r", vmin=-4.5, vmax=-2.1)
    fig.colorbar(pcm1, ax=ax1, label=r"${\rm lgFburst}$")
    for ax in axes:
        xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    ylabel = ax0.set_ylabel(r"${\rm \log_{10}sSFR}$")

    ax0.set_title(label1)
    ax1.set_title(label2)

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig


def _lgfburst_weighted(arr):
    arr = np.asarray(arr)

    m_star = 10 ** arr[:, 0]
    f_burst = 10 ** arr[:, 1]
    p_burst = arr[:, 2]
    weight = arr[:, 3]

    m_star_burst = np.sum(m_star * f_burst * p_burst * weight)
    m_star_tot = np.sum(m_star * weight)

    lgfburst = np.log10(m_star_burst / m_star_tot)

    return lgfburst


def plot_lgfburst_smooth(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    num_halos=10000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    gridsize=40,
    mincnt=0.5,
    sigma=1.0,
    plot="cen+sat",
    plt_show=True,
):
    (
        lc_data,
        phot_data,
        gal_weight,
        phot_randoms,
        merging_randoms,
    ) = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        mc_merge=1,
    )

    if plot == "cen":
        sel = lc_data.is_central == 1
    elif plot == "sat":
        sel = lc_data.is_central != 1
    elif plot == "cen+sat":
        sel = np.isfinite(lc_data.is_central)

    sel &= phot_data.p_merge < 0.5

    logsfr_10Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.01)
    logsm_obs = phot_data.logsm_obs
    logssfr_10 = logsfr_10Myr - logsm_obs

    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_data.logsm_obs[sel],
        logssfr_10[sel],
    )
    C = np.column_stack(
        [
            logsm_obs[sel],
            phot_data.lgfburst[sel],
            p_burst,
            gal_weight[sel],
        ]
    )
    fig_width = 7.1
    fig_height = 3.05
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.01)
    labelsize = 10
    fontsize = 10
    labelsize = 9
    vmin, vmax = 10**-6, 10**-1.5
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sm_limits = (8, 12)
    hm_limits = (lgmp_min, lgmp_max)

    """Plot fburst w/ halo mass and redshift"""
    x, y = lc_data.z_obs[sel], lc_data.logmp_obs[sel]
    bins = (gridsize, gridsize)
    rng = [[z_min, z_max], [hm_limits[0], hm_limits[1]]]

    w = C[:, 3]
    m_star = 10 ** C[:, 0]
    wm = m_star * w

    num, xe, ye = np.histogram2d(
        x, y, bins=bins, range=rng, weights=wm * 10 ** C[:, 1] * C[:, 2]
    )
    den, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=wm)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    img[(gaussian_filter(cnt, sigma) < mincnt)] = np.nan

    ax[0].pcolormesh(xe, ye, img.T, cmap="coolwarm_r", norm=norm, rasterized=True)

    ax[0].set_ylabel(r"log$_{10}$ (M$_{h}$ [M${_\odot}$])", fontsize=fontsize)
    ax[0].set_ylim(hm_limits)

    """Plot fburst w/ stellar mass and redshift"""
    x, y = lc_data.z_obs[sel], logsm_obs[sel]
    bins = (gridsize, gridsize)
    rng = [[z_min, z_max], [sm_limits[0], sm_limits[1]]]

    w = C[:, 3]
    m_star = 10 ** C[:, 0]
    wm = m_star * w

    num, xe, ye = np.histogram2d(
        x, y, bins=bins, range=rng, weights=wm * 10 ** C[:, 1] * C[:, 2]
    )
    den, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=wm)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    img[(gaussian_filter(cnt, sigma) < mincnt)] = np.nan

    ax1_c = ax[1].pcolormesh(
        xe, ye, img.T, cmap="coolwarm_r", norm=norm, rasterized=True
    )

    cbar = fig.colorbar(
        ax1_c,
        ax=ax,
        pad=0.005,
        extend="both",
        label=r"f$_{*,\ recent\ burst}$ $\equiv$ M$_{*,\ recent\ burst}$ / M$_{*,\ total}$",
    )
    cbar.ax.tick_params(labelsize=labelsize, direction="in")

    ax[1].set_ylabel(r"log$_{10}$ (M$_{*}$ [M${_\odot}$])", fontsize=fontsize)
    ax[1].set_ylim(sm_limits)

    for i in range(0, 2):
        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        ax[i].set_xlim(z_min, z_max)
        ax[i].set_xlabel("redshift", fontsize=fontsize)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))
    fig.savefig(
        savedir
        + "/"
        + model_nickname
        + "_fburst_mh_z"
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


def plot_lgfburst_ha_uv(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    line_wave_table_ha_uv,
    run_label,
    savedir,
    num_halos=10000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    gridsize=40,
    mincnt=0.5,
    sigma=1.0,
    vmin=10**-6.0,
    vmax=10**-1.5,
    plot="cen+sat",
    plt_show=True,
):
    phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        line_wave_table_ha_uv,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        mc_merge=1,
    )
    gal_weight = lc_data.halo_weight
    lgL_halpha = np.log10(photline_data.linelum_gal[:, 0])
    M_uv = compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 1])

    if plot == "cen":
        sel = lc_data.is_central == 1
    elif plot == "sat":
        sel = lc_data.is_central != 1
    elif plot == "cen+sat":
        sel = np.isfinite(lc_data.is_central)

    sel &= phot_data.p_merge < 0.5

    logsfr_10Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.01)
    logsm_obs = phot_data.logsm_obs
    logssfr_10 = logsfr_10Myr - logsm_obs

    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_data.logsm_obs[sel],
        logssfr_10[sel],
    )
    C = np.column_stack(
        [
            logsm_obs[sel],
            phot_data.lgfburst[sel],
            p_burst,
            gal_weight[sel],
        ]
    )
    fig_width = 7.1
    fig_height = 3.05
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.01)
    labelsize = 10
    fontsize = 10
    labelsize = 9

    norm = LogNorm(vmin=vmin, vmax=vmax)
    uv_limits = (-22, -16)
    ha_limits = (40, 43.5)

    """Plot fburst w/ halpha and redshift"""
    x, y = lc_data.z_obs[sel], lgL_halpha[sel]
    bins = (gridsize, gridsize)
    rng = [[z_min, z_max], [ha_limits[0], ha_limits[1]]]

    w = C[:, 3]
    m_star = 10 ** C[:, 0]
    wm = m_star * w

    num, xe, ye = np.histogram2d(
        x, y, bins=bins, range=rng, weights=wm * 10 ** C[:, 1] * C[:, 2]
    )
    den, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=wm)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    img[(gaussian_filter(cnt, sigma) < mincnt)] = np.nan

    ax[0].pcolormesh(xe, ye, img.T, cmap="coolwarm_r", norm=norm, rasterized=True)

    ax[0].set_ylabel(r"log$_{10}$ (L$_{H\alpha}$ [erg s$^{-1}$])", fontsize=fontsize)
    ax[0].set_ylim(ha_limits)

    """Plot fburst w/ UV and redshift"""
    x, y = lc_data.z_obs[sel], M_uv[sel]
    bins = (gridsize, gridsize)
    rng = [[z_min, z_max], [uv_limits[0], uv_limits[1]]]

    w = C[:, 3]
    m_star = 10 ** C[:, 0]
    wm = m_star * w

    num, xe, ye = np.histogram2d(
        x, y, bins=bins, range=rng, weights=wm * 10 ** C[:, 1] * C[:, 2]
    )
    den, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=wm)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    img[(gaussian_filter(cnt, sigma) < mincnt)] = np.nan

    ax1_c = ax[1].pcolormesh(
        xe, ye, img.T, cmap="coolwarm_r", norm=norm, rasterized=True
    )

    cbar = fig.colorbar(
        ax1_c,
        ax=ax,
        pad=0.005,
        extend="both",
        label=r"f$_{*,\ recent\ burst}$ $\equiv$ M$_{*,\ recent\ burst}$ / M$_{*,\ total}$",
    )
    cbar.ax.tick_params(labelsize=labelsize, direction="in")

    ax[1].set_ylabel(r"M$_{UV}$", fontsize=fontsize)
    ax[1].set_ylim(uv_limits[::-1])

    for i in range(0, 2):
        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        ax[i].set_xlim(z_min, z_max)
        ax[i].set_xlabel("redshift", fontsize=fontsize)

    fig.savefig(
        os.path.join(savedir, f"{run_label}_fburst_ha_uv_{z_min:.2f}-{z_max:.2f}.png"),
        bbox_inches="tight",
        dpi=200,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_lgfburst_ha(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    line_wave_table_ha,
    run_label,
    savedir,
    num_halos=10000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    gridsize=40,
    mincnt=0.5,
    sigma=1.0,
    vmin=10**-6.0,
    vmax=10**-1.5,
    vmin_comp=9.9e-1,
    vmax_comp=1e0,
    plot="cen+sat",
    plt_show=True,
):
    phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        line_wave_table_ha,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        mc_merge=1,
    )
    gal_weight = lc_data.halo_weight
    lgL_halpha = np.log10(photline_data.linelum_gal[:, 0])

    if plot == "cen":
        sel = lc_data.is_central == 1
    elif plot == "sat":
        sel = lc_data.is_central != 1
    elif plot == "cen+sat":
        sel = np.isfinite(lc_data.is_central)

    sel &= phot_data.p_merge < 0.5

    logsfr_10Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.01)
    logsm_obs = phot_data.logsm_obs
    logssfr_10 = logsfr_10Myr - logsm_obs

    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_data.logsm_obs[sel],
        logssfr_10[sel],
    )
    C = np.column_stack(
        [
            logsm_obs[sel],
            phot_data.lgfburst[sel],
            p_burst,
            gal_weight[sel],
        ]
    )

    x, y = lc_data.z_obs[sel], lgL_halpha[sel]
    ha_limits = (40, 43.5)
    bins = (gridsize, gridsize)
    rng = [[z_min, z_max], [ha_limits[0], ha_limits[1]]]

    wm = 10 ** C[:, 0] * C[:, 3]

    num, xe, ye = np.histogram2d(
        x, y, bins=bins, range=rng, weights=wm * 10 ** C[:, 1] * C[:, 2]
    )
    den, _, _ = np.histogram2d(x, y, bins=bins, range=rng, weights=wm)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        img = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    img[gaussian_filter(cnt, sigma) < mincnt] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01)
    labelsize = 9
    fontsize = 10

    m0 = ax[0].pcolormesh(
        xe,
        ye,
        img.T,
        cmap="coolwarm_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    m1 = ax[1].pcolormesh(
        xe,
        ye,
        (1 - img).T,
        cmap="coolwarm",
        norm=LogNorm(vmin=vmin_comp, vmax=vmax_comp),
        rasterized=True,
    )

    cb0 = fig.colorbar(
        m0,
        ax=ax[0],
        pad=0.005,
        extend="both",
        label=r"f$_{*,\ recent\ burst}$",
    )
    cb1 = fig.colorbar(
        m1,
        ax=ax[1],
        pad=0.005,
        extend="min",
        label=r"1 - f$_{*,\ recent\ burst}$",
    )
    for cb in (cb0, cb1):
        cb.ax.tick_params(labelsize=labelsize, direction="in")

    for a in ax:
        a.minorticks_on()
        a.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        a.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        a.set_xlim(z_min, z_max)
        a.set_ylim(ha_limits)
        a.set_xlabel("redshift", fontsize=fontsize)
        a.set_ylabel(r"log$_{10}$ (L$_{H\alpha}$ [erg s$^{-1}$])", fontsize=fontsize)

    fig.savefig(
        os.path.join(savedir, f"{run_label}_fburst_ha_{z_min:.2f}-{z_max:.2f}.png"),
        bbox_inches="tight",
        dpi=200,
    )

    if plt_show:
        plt.show()
    plt.close()


def plot_lgfburst_hexbin(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    num_halos=10000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    gridsize=40,
    mincnt=1,
    plot="cen+sat",
    plt_show=True,
):
    (
        lc_data,
        phot_data,
        gal_weight,
        phot_randoms,
        merging_randoms,
    ) = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        mc_merge=1,
    )

    if plot == "cen":
        sel = lc_data.is_central == 1
    elif plot == "sat":
        sel = lc_data.is_central != 1
    elif plot == "cen+sat":
        sel = np.isfinite(lc_data.is_central)

    logsfr_10Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.01)
    logsm_obs = phot_data.logsm_obs
    logssfr_10 = logsfr_10Myr - logsm_obs

    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_data.logsm_obs,
        logssfr_10,
    )
    C = np.column_stack(
        [
            logsm_obs[sel],
            phot_data.lgfburst[sel],
            p_burst[sel],
            gal_weight[sel],
        ]
    )
    fig_width = 7.1
    fig_height = 3.05
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    labelsize = 10
    fontsize = 10
    labelsize = 9
    vmin, vmax = -6, -1.5
    xticks = [0.02, 0.5, 1.0, 1.5, 2.0, 2.5]
    xticklabels = ["0.02", "0.5", "1.0", "1.5", "2.0", "2.5"]

    sm_yticks = [8, 9, 10, 11, 12]
    sm_yticklabels = ["8", "9", "10", "11", "12"]
    sm_limits = (8, 12)

    hm_yticks = [10, 11, 12, 13, 14, 15]
    hm_yticklabels = ["10", "11", "12", "13", "14", "15"]
    hm_limits = (lgmp_min, lgmp_max)

    """Plot fburst w/ halo mass and redshift"""
    ax[0].hexbin(
        lc_data.z_obs[sel],
        lc_data.logmp_obs[sel],
        C=C,
        reduce_C_function=_lgfburst_weighted,
        cmap="coolwarm_r",
        vmin=vmin,
        vmax=vmax,
        mincnt=mincnt,
        extent=(z_min, z_max, hm_limits[0], hm_limits[1]),
        gridsize=gridsize,
        rasterized=True,
    )

    ax[0].set_ylabel(r"log$_{10}$ (M$_{h}$ [M${_\odot}$])", fontsize=fontsize)
    ax[0].set_ylim(hm_limits)
    ax[0].set_yticks(hm_yticks)
    ax[0].set_yticklabels(hm_yticklabels)

    """Plot fburst w/ stellar mass and redshift"""
    hb1 = ax[1].hexbin(
        lc_data.z_obs[sel],
        logsm_obs[sel],
        C=C,
        reduce_C_function=_lgfburst_weighted,
        cmap="coolwarm_r",
        vmin=vmin,
        vmax=vmax,
        mincnt=mincnt,
        extent=(z_min, z_max, sm_limits[0], sm_limits[1]),
        gridsize=gridsize,
        rasterized=True,
    )

    cbar = fig.colorbar(
        hb1,
        ax=ax[1],
        label="log$_{10}$ (fraction of stellar mass formed in a recent burst)",
    )
    cbar.ax.tick_params(labelsize=labelsize, direction="in")

    ax[1].set_ylabel(r"log$_{10}$ (M$_{*}$ [M${_\odot}$])", fontsize=fontsize)
    ax[1].set_ylim(sm_limits)
    ax[1].set_yticks(sm_yticks)
    ax[1].set_yticklabels(sm_yticklabels)

    for i in range(0, 2):
        ax[i].minorticks_on()
        ax[i].tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=6,
            width=1,
            labelsize=labelsize,
        )
        ax[i].tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticklabels)
        ax[i].set_xlim(z_min, z_max)
        ax[i].set_xlabel("redshift", fontsize=fontsize)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))
    fig.savefig(
        savedir
        + "/"
        + model_nickname
        + "_fburst_mh_z"
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
