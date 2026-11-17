import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..kernels.lc_phot_kern import multiband_lc_phot_kern
from ..kernels.sfr_tau import get_logsfr_tau
from ..tab_blue_orange_cmap import make_cmap
from .plot_utils import make_thresholded_reduce_C_function, percentile_norm

plt.rc("font", family="serif", serif=["Times New Roman"])
cmap = make_cmap()


def plot_burst_ratio(
    ran_key,
    param_collection,
    zbins,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    short_tau_myr=10,
    long_tau_myr=50,
    num_halos=1000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    gridsize=(60, 60),
    mincnt=1,
    xlim=(8, 12),
    ylim=(-1, 1),
    plt_show=True,
):
    n_zbins = len(zbins)
    fig, ax = plt.subplots(
        1,
        n_zbins,
        figsize=(7.1, 2.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0},
        squeeze=False,
    )
    ax = ax[0]
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.2, top=0.9)
    labelsize = 9
    fontsize = 10
    hbs = []

    short_tau_gyr = short_tau_myr / 1e3
    long_tau_gyr = long_tau_myr / 1e3

    for z in range(n_zbins):
        z_min, z_max = zbins[z]
        ax[z].set_title(f"{z_min}<z<{z_max}", fontsize=fontsize)
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
        logsm_obs = phot_data.logsm_obs
        unmerged = phot_data.p_merge < 0.5

        logsfr_short = get_logsfr_tau(
            phot_data, lc_data, ssp_data, tau_gyr=short_tau_gyr
        )
        logsfr_long = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=long_tau_gyr)

        sfr_short = 10**logsfr_short
        sfr_long = 10**logsfr_long
        sfr_short_long = (long_tau_myr * sfr_long - short_tau_myr * sfr_short) / (
            long_tau_myr - short_tau_myr
        )
        logsfr_short_long = np.log10(sfr_short_long)

        b_short_long = logsfr_short - logsfr_short_long

        sel = (
            (logsm_obs > xlim[0])
            & (logsm_obs < xlim[1])
            & (b_short_long > ylim[0])
            & (b_short_long < ylim[1])
            & unmerged
        )

        reduce_C_function = make_thresholded_reduce_C_function(
            gal_weight[unmerged], threshold=1e-6
        )

        hb = ax[z].hexbin(
            logsm_obs[sel],
            b_short_long[sel],
            C=gal_weight[sel],
            reduce_C_function=reduce_C_function,
            cmap=cmap,
            gridsize=gridsize,
            mincnt=mincnt,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        )
        hbs.append(hb)

        ax[z].axhline(0.2, ls=":", c="k", alpha=0.5)
        ax[z].axhline(-0.2, ls=":", c="k", alpha=0.5)

        ax[z].minorticks_on()
        ax[z].xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
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

    norm = percentile_norm([hb.get_array() for hb in hbs], lo=0.001, hi=99.9)
    for hb in hbs:
        hb.set_norm(norm)

    cax = fig.add_axes([0.915, 0.2, 0.012, 0.7])
    cb = fig.colorbar(hbs[0], cax=cax)
    cb.set_label("f", fontsize=fontsize)
    cb.ax.tick_params(labelsize=labelsize)

    fig.supxlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel(
        rf"$b_{{{short_tau_myr}-{long_tau_myr}}} \equiv \log_{{10}}\left(\mathrm{{SFR}}_{{{short_tau_myr}}}/\mathrm{{SFR}}_{{{short_tau_myr}-{long_tau_myr}}}\right)$",
        fontsize=fontsize,
    )
    fig.savefig(
        f"{savedir}/{run_label}_b_{short_tau_myr}-{long_tau_myr}_hexbin.png", dpi=600
    )

    if plt_show:
        plt.show()
    plt.close()
