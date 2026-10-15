import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..kernels.line_dustfree import linelum_gal_dustfree
from ..lightcone_generators import generate_lc_data
from ..tab_blue_orange_cmap import make_cmap
from .plot_utils import make_thresholded_reduce_C_function, percentile_norm

cmap = make_cmap()
plt.rc("font", family="serif", serif=["Times New Roman"])


COLORS_Z = ["#2d0b52", "#0491a1", "#a1d661", "#ee6920", "#8a0f17"]

LOGHUV_MIN, LOGHUV_MAX = -3.5, -0.5
LOGMP_OBS_MIN, LOGMP_OBS_MAX = 10.0, 13.0
LOGSM_OBS_MIN, LOGSM_OBS_MAX = 7.0, 12.5


def plot_halpha_uv_ratio(
    ran_key,
    z_min,
    z_max,
    param_collection,
    line_wave_table,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    n_z_phot_table=15,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    num_halos=1000,
    plt_show=True,
):
    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    dustfree_linelum_gal, phot_data = linelum_gal_dustfree(
        ran_key,
        lc_data,
        ssp_data,
        line_wave_table,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
    )

    log_halpha_uv = np.log10(dustfree_linelum_gal[:, 0] / dustfree_linelum_gal[:, 1])

    low_mass = (phot_data.logsm_obs > 7.0) & (phot_data.logsm_obs < 8.0)
    intermediate_mass = (phot_data.logsm_obs > 8.0) & (phot_data.logsm_obs < 9.0)
    high_mass = phot_data.logsm_obs > 9.0

    unmerged = phot_data.p_merge < 0.5

    bins = np.arange(-2.25, -1.4, 0.05)
    titlesize = 11
    labelsize = 11
    legendsize = 8
    alpha = 0.6
    zbin_edges = np.arange(z_min, z_max + 1, 1)
    n_zbins = len(zbin_edges) - 1

    fig, ax = plt.subplots(1, n_zbins, figsize=(7.1, 2.5), constrained_layout=True)

    for zbin in range(n_zbins):
        z_mask = (lc_data.z_obs > zbin_edges[zbin]) & (
            lc_data.z_obs < zbin_edges[zbin + 1]
        )

        z_med = str(np.median([zbin_edges[zbin], zbin_edges[zbin + 1]]))
        ax[zbin].set_title(r"$z = $" + z_med, fontsize=titlesize)

        ax[zbin].axvspan(-1.93, -1.78, alpha=0.1, color="k", label="Mehta+ 2023")

        ax[zbin].hist(
            log_halpha_uv[unmerged & z_mask & low_mass],
            weights=gal_weight[unmerged & z_mask & low_mass],
            bins=bins,
            color="#A7C7E7",
            alpha=alpha,
            label=r"7 < log$_{10}$(M$_{*}$/M$_{\odot}$) < 8",
            density=True,
        )
        ax[zbin].hist(
            log_halpha_uv[unmerged & z_mask & intermediate_mass],
            weights=gal_weight[unmerged & z_mask & intermediate_mass],
            bins=bins,
            color="#A7C7A0",
            alpha=alpha,
            label=r"8 < log$_{10}$(M$_{*}$/M$_{\odot}$) < 9",
            density=True,
        )
        ax[zbin].hist(
            log_halpha_uv[unmerged & z_mask & high_mass],
            weights=gal_weight[unmerged & z_mask & high_mass],
            bins=bins,
            color="#E57373",
            alpha=alpha,
            label=r"log$_{10}$(M$_{*}$/M$_{\odot}$) > 9",
            density=True,
        )

        ax[zbin].tick_params(
            which="major",
            direction="in",
            top=True,
            right=False,
            length=6,
            width=1,
            labelsize=labelsize,
        )

        ax[zbin].minorticks_on()
        ax[zbin].tick_params(
            which="minor",
            direction="in",
            top=True,
            length=3,
            width=0.8,
            labelsize=labelsize,
        )

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=legendsize)
    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ / L$_{UV}$)", fontsize=labelsize)
    fig.supylabel("PDF", fontsize=labelsize)

    fig.savefig(
        savedir + "/" + run_label + "_halpha_uv_ratio" + ".png",
        dpi=400,
    )

    if plt_show:
        plt.show()

    plt.close()


def plot_halpha_uv_ratio_mass_z(
    ran_key,
    z_min,
    z_max,
    param_collection,
    line_wave_table,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    num_halos=1000,
    gridsize=(50, 25),
    threshold=1e-6,
    plt_show=True,
):
    labelsize = 11
    legendsize = 10
    z_phot_table = 10 ** jnp.linspace(jnp.log10(z_min), jnp.log10(z_max), 15)
    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight
    dustfree_linelum_gal, phot_data = linelum_gal_dustfree(
        ran_key,
        lc_data,
        ssp_data,
        line_wave_table,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
    )

    log_halpha_uv = np.log10(dustfree_linelum_gal[:, 0] / dustfree_linelum_gal[:, 1])
    unmerged = phot_data.p_merge < 0.5

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5), constrained_layout=True)
    reduce_C_function = make_thresholded_reduce_C_function(
        gal_weight[unmerged], threshold
    )

    hb = ax[0].hexbin(
        lc_data.logmp_obs[unmerged],
        log_halpha_uv[unmerged],
        C=gal_weight[unmerged],
        reduce_C_function=reduce_C_function,
        cmap=cmap,
        gridsize=gridsize,
        extent=(LOGMP_OBS_MIN, LOGMP_OBS_MAX, LOGHUV_MIN, LOGHUV_MAX),
    )
    hb1 = ax[1].hexbin(
        phot_data.logsm_obs[unmerged],
        log_halpha_uv[unmerged],
        C=gal_weight[unmerged],
        reduce_C_function=reduce_C_function,
        cmap=cmap,
        gridsize=gridsize,
        extent=(LOGSM_OBS_MIN, LOGSM_OBS_MAX, LOGHUV_MIN, LOGHUV_MAX),
    )
    hb2 = ax[2].hexbin(
        lc_data.z_obs[unmerged],
        log_halpha_uv[unmerged],
        C=gal_weight[unmerged],
        reduce_C_function=reduce_C_function,
        cmap=cmap,
        gridsize=gridsize,
        extent=(z_min, z_max, LOGHUV_MIN, LOGHUV_MAX),
    )

    # shared norm computed from both panels' counts
    counts = hb.get_array()
    counts1 = hb1.get_array()
    counts2 = hb2.get_array()
    norm = percentile_norm([counts, counts1, counts2])
    hb.set_norm(norm)
    hb1.set_norm(norm)
    hb2.set_norm(norm)

    cbar2 = fig.colorbar(hb2, ax=ax[2])
    cbar2.set_label("$\U0001D453$", fontsize=labelsize)
    cbar2.ax.tick_params(labelsize=labelsize)

    for i in range(0, 3):
        ax[i].axhspan(-1.93, -1.78, alpha=0.2, color="k", label="Mehta+ 2023")
        ax[i].set_ylim(LOGHUV_MIN, LOGHUV_MAX)
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
    ax[0].set_xlim(LOGMP_OBS_MIN, LOGMP_OBS_MAX)
    ax[1].set_xlim(LOGSM_OBS_MIN, LOGSM_OBS_MAX)
    ax[2].set_xlim(z_min, z_max)
    ax[-1].legend(fontsize=legendsize)
    ax[0].set_xlabel(r"log$_{10}$ (M$_{h}$ [M$_{\odot}$])", fontsize=labelsize)
    ax[1].set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=labelsize)
    ax[2].set_xlabel("redshift", fontsize=labelsize)
    fig.supylabel("log$_{10}$ (L$_{H\u03b1}$ / L$_{UV}$)", fontsize=labelsize)
    fig.savefig(
        savedir + "/" + run_label + "_halpha_uv_ratio_mass" + ".png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()
