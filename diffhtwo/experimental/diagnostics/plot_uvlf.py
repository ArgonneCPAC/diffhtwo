import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from ..kernels.lc_photline_kern import mc_photline_kern_merging_wrapper
from ..kernels.line_dustfree import linelum_gal_dustfree
from ..lc_utils import zbin_vol
from ..lightcone_generators import generate_lc_data
from ..tab_blue_orange_cmap import make_cmap
from ..uv_luminosity import UV_FREQUENCY_HZ
from .plot_phot import minerva_colors

cmap = make_cmap()
plt.rc("font", family="serif", serif=["Times New Roman"])


COLORS_Z = ["#2d0b52", "#0491a1", "#a1d661", "#ee6920", "#8a0f17"]

LOGHUV_MIN, LOGHUV_MAX = -3.5, -0.5
LOGMP_OBS_MIN, LOGMP_OBS_MAX = 10.0, 13.0
LOGSM_OBS_MIN, LOGSM_OBS_MAX = 7.0, 12.5

# copied from astropy import units as u: d = (10 * u.pc).to(u.cm).value
D_10pc_in_cm = 3.085677581491367e19


def _compute_M_UV_from_L_UV(L_UV, D_10pc_in_cm=D_10pc_in_cm):
    """
    Input:
        L_UV: UV Luminosity in erg/s
    """

    f_nu = L_UV / UV_FREQUENCY_HZ / (4 * np.pi * D_10pc_in_cm**2)
    M_UV = -2.5 * np.log10(f_nu) - 48.6

    return M_UV


def plot_uvlf(
    ran_key,
    param_collection,
    line_wave_table_uv,
    ssp_data_uv,
    tcurves,
    run_label,
    savedir,
    bouwens21_drn,
    n_z_phot_table=15,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    num_halos=1000,
    bin_width=0.5,
    plt_show=True,
    cosmo_params=DEFAULT_COSMOLOGY,
    mc_merge=1,
):
    z_min, z_max = 1.5, 6.5
    zbin_edges = np.arange(z_min, z_max + 1, 1)
    z_centers = 0.5 * (zbin_edges[1:] + zbin_edges[:-1])

    n_zbins = len(zbin_edges) - 1
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
        ssp_data_uv,
        tcurves,
        z_phot_table,
        cosmo_params=cosmo_params,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    # Dust-free
    dustfree_linelum_gal, dustfree_phot_data = linelum_gal_dustfree(
        ran_key,
        lc_data,
        ssp_data_uv,
        line_wave_table_uv,
        param_collection.diffstarpop_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        mc_merge=mc_merge,
    )
    dustfree_unmerged = dustfree_phot_data.p_merge < 0.5
    dustfree_M_uv = _compute_M_UV_from_L_UV(dustfree_linelum_gal[:, 0])

    # Dust-obscured
    phot_data, phot_randoms, photline_data = mc_photline_kern_merging_wrapper(
        ran_key,
        param_collection,
        lc_data,
        line_wave_table_uv,
        mc_merge=mc_merge,
    )
    M_uv = _compute_M_UV_from_L_UV(photline_data.linelum_gal[:, 0])
    unmerged = phot_data.p_merge < 0.5

    labelsize = 11
    legendsize = 7
    alpha = 0.6
    colors = minerva_colors(n_zbins)
    bin_edges = np.arange(-23, -15, bin_width)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    fig, ax = plt.subplots(1, figsize=(3.5, 3.5), constrained_layout=True)

    for zbin in range(n_zbins):
        bouwens_uvlf = np.loadtxt(
            f"{bouwens21_drn}/uvlf_z{str(int(z_centers[zbin]))}.txt"
        )
        ax.errorbar(
            bouwens_uvlf[:, 0],
            bouwens_uvlf[:, 1],
            bouwens_uvlf[:, 2],
            color=colors[zbin],
            fmt="s",
            alpha=0.8,
            ms=4,
        )

        z_min_current = zbin_edges[zbin]
        z_max_current = zbin_edges[zbin + 1]

        z_mask = (lc_data.z_obs > z_min_current) & (lc_data.z_obs < z_max_current)

        z_med = "z=" + str(np.median([z_min_current, z_max_current]))

        lc_tot_vol_mpc3 = zbin_vol(
            lc_data.sky_area_degsq, z_min_current, z_max_current, cosmo_params
        )
        dustfree_phi, _ = np.histogram(
            dustfree_M_uv[dustfree_unmerged & z_mask],
            weights=gal_weight[dustfree_unmerged & z_mask]
            * (1 / lc_tot_vol_mpc3)
            * (1 / bin_width),
            bins=bin_edges,
        )
        ax.plot(bin_centers, dustfree_phi, color=colors[zbin], alpha=0.3, ls="--")

        phi, _ = np.histogram(
            M_uv[unmerged & z_mask],
            weights=gal_weight[unmerged & z_mask]
            * (1 / lc_tot_vol_mpc3)
            * (1 / bin_width),
            bins=bin_edges,
        )
        ax.plot(
            bin_centers,
            phi,
            color=colors[zbin],
            alpha=0.8,
            label=z_med,
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 5e-2)
    ax.set_xlim(-23, -16)
    ax.minorticks_on()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        left=True,
        bottom=True,
        length=6,
        width=1,
        labelsize=labelsize,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        left=True,
        bottom=True,
        length=3,
        width=0.8,
        labelsize=labelsize,
    )

    handles, labels = ax.get_legend_handles_labels()
    handle_bouwens21 = ax.errorbar(
        [],
        [],
        yerr=[[0.2], [0.2]],
        fmt="s",
        color="k",
        markersize=6,
        linestyle="none",
        lw=1,
    )
    handle_dustfree = ax.plot([], [], color="k", ls="--", alpha=alpha)[0]

    fig.legend(
        handles + [handle_bouwens21, handle_dustfree],
        labels + ["Bouwens+21", "Dust-free"],
        loc="outside upper center",
        ncol=4,
        fontsize=legendsize,
    )

    ax.set_xlabel(r"M$_{UV, 1600\AA}$ [AB]", fontsize=labelsize)
    ax.set_ylabel(r"$\phi$ [mag$^{-1}$ Mpc$^{-3}$]", fontsize=labelsize)

    fig.savefig(
        savedir + "/" + run_label + "_uvlf" + ".png",
        dpi=400,
    )

    if plt_show:
        plt.show()

    plt.close()
