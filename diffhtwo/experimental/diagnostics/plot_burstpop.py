"""
Based on diffsky.burstpop.diagnostics.plot_fburstpop
"""
import numpy as np
from diffsky.burstpop import freqburst_mono
from diffsky.burstpop.fburstpop_mono import get_fburst_from_fburstpop_params
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from ..kernels.lc_phot_kern import multiband_lc_phot_kern

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
    gal_weight = arr[:, 3]

    m_star_burst = np.sum(m_star * f_burst * p_burst * gal_weight)
    m_star_tot = np.sum(m_star * gal_weight)

    lgfburst = np.log10(m_star_burst / m_star_tot)

    return lgfburst


def plot_lgfburst_mh_z(
    ran_key,
    param_collection,
    z_min,
    z_max,
    ssp_data,
    tcurves,
    model_nickname,
    savedir,
    mag_thresh=None,
    frac_cat=None,
    num_halos=10000,
    gridsize=40,
    mincnt=1,
    plot="cen+sat",
    plt_show=True,
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

    if plot == "cen":
        sel = lc_data.is_central == 1
    elif plot == "sat":
        sel = lc_data.is_central != 1
    elif plot == "cen+sat":
        sel = np.isfinite(lc_data.is_central)

    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        param_collection.spspop_params.burstpop_params.freqburst_params,
        phot_kern_results.logsm_obs,
        phot_kern_results.logssfr_obs,
    )
    C = np.column_stack(
        [
            phot_kern_results.logsm_obs[sel],
            phot_kern_results.lgfburst[sel],
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
    hm_limits = (10, 15)

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
        phot_kern_results.logsm_obs[sel],
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

    cbar = fig.colorbar(hb1, ax=ax[1], label="log$_{10}$ (burst fraction)")
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
