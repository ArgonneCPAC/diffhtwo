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
    gridsize=25,
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
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[1, 1.2])
    vmin, vmax = -6, -1.5

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
        gridsize=gridsize,
        rasterized=True,
    )

    ax[0].set_xlabel("redshift")
    ax[0].set_ylabel("log$_{10}$ (M$_{h, peak}$ [M\u2609])")
    ax[0].set_xlim(z_min, z_max)
    ax[0].set_ylim(10, 15)

    """Plot fburst w/ stellar mass and redshift"""
    logsm_min, log_sm_max = 8, 12
    hb1 = ax[1].hexbin(
        lc_data.z_obs[sel],
        phot_kern_results.logsm_obs[sel],
        C=C,
        reduce_C_function=_lgfburst_weighted,
        cmap="coolwarm_r",
        vmin=vmin,
        vmax=vmax,
        mincnt=mincnt,
        extent=(z_min, z_max, logsm_min, log_sm_max),
        gridsize=gridsize,
        rasterized=True,
    )
    cbar1 = plt.colorbar(hb1, ax=ax[1], label="log$_{10}$ ($\U0001D453_{burst}$)")
    cbar1.ax.invert_yaxis()

    ax[1].set_xlabel("redshift")
    ax[1].set_ylabel("log$_{10}$ (M$_{*}$ [M\u2609])")
    ax[1].set_xlim(z_min, z_max)
    ax[1].set_ylim(logsm_min, log_sm_max)

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
