import matplotlib.pyplot as plt
import numpy as np

from ..kernels.lc_phot_kern import multiband_lc_phot_kern
from ..kernels.smhm import get_ex_situ_frac_median_v_sm

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_ex_situ_frac_z0(
    ran_key,
    param_collection,
    num_halos,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    lit_drn,
    logmp_obs_min=10.0,
    logmp_obs_max=15.0,
    logsm_obs_min=8.5,
    logsm_obs_max=12.5,
    d_dex=0.15,
    plt_show=True,
):
    z_min, z_max = 0.02, 0.05

    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )

    """ex-situ frac (halo mass)"""
    # logmp_bins = np.arange(logmp_obs_min, logmp_obs_max + d_dex, d_dex)
    # logmp_bin_centers = (logmp_bins[:-1] + logmp_bins[1:]) / 2
    # ex_situ_frac_median_hm = get_ex_situ_frac_median_v_hm(
    #     logmp_bins,
    #     lc_data.logmp_obs,
    #     phot_data.logsm_obs,
    #     phot_data.logsm_obs_in_situ,
    #     gal_weight,
    #     lc_data.is_central,
    # )

    """ex-situ frac (stellar mass)"""
    logsm_bins = np.arange(logsm_obs_min, logsm_obs_max + d_dex, d_dex)
    logsm_bin_centers = (logsm_bins[:-1] + logsm_bins[1:]) / 2

    ex_situ_frac_median_sm = get_ex_situ_frac_median_v_sm(
        logsm_bins,
        phot_data.logsm_obs,
        phot_data.logsm_obs_in_situ,
        gal_weight,
        lc_data.is_central,
    )

    fig_width = 3.5
    fig_height = 3.6
    fig, ax = plt.subplots(
        1,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw={"wspace": 0},
    )
    titlesize = 12
    fontsize = 11
    labelsize = 11
    legendsize = 8
    alpha = 0.75

    ax.plot(logsm_bin_centers, ex_situ_frac_median_sm, c="k", label="diffsky")

    """Literature"""
    davison2020 = np.loadtxt(lit_drn + "/Davison2020.csv", delimiter=",")
    ax.plot(
        np.log10(davison2020[:, 0]),
        davison2020[:, 1],
        c="tab:orange",
        alpha=alpha,
        label="Davison+2020 (EAGLE)",
    )

    davison2020_lo = np.loadtxt(lit_drn + "/Davison2020_lo.csv", delimiter=",")
    davison2020_hi = np.loadtxt(lit_drn + "/Davison2020_hi.csv", delimiter=",")
    davison2020_x = np.linspace(
        max(davison2020_lo[:, 0].min(), davison2020_hi[:, 0].min()),
        min(davison2020_lo[:, 0].max(), davison2020_hi[:, 0].max()),
        5000,
    )
    davison2020_lo_y = np.interp(
        davison2020_x, davison2020_lo[:, 0], davison2020_lo[:, 1]
    )
    davison2020_hi_y = np.interp(
        davison2020_x, davison2020_hi[:, 0], davison2020_hi[:, 1]
    )
    ax.fill_between(
        np.log10(davison2020_x),
        davison2020_lo_y,
        davison2020_hi_y,
        alpha=0.2,
        color="tab:orange",
    )

    tachella2019 = np.loadtxt(lit_drn + "/Tachella2019.csv", delimiter=",")
    ax.plot(
        np.log10(tachella2019[:, 0]),
        tachella2019[:, 1],
        c="deepskyblue",
        alpha=alpha,
        label="Tachella+2019 (IllustrisTNG)",
    )

    rodriguez2016 = np.loadtxt(lit_drn + "/Rodriguez2016.csv", delimiter=",")
    ax.plot(
        np.log10(rodriguez2016[:, 0]),
        rodriguez2016[:, 1],
        c="tab:green",
        alpha=alpha,
        label="Rodriguez+2016 (Illustris-1)",
    )

    rodriguez2016_lo = np.loadtxt(lit_drn + "/Rodriguez2016_lo.csv", delimiter=",")
    rodriguez2016_hi = np.loadtxt(lit_drn + "/Rodriguez2016_hi.csv", delimiter=",")
    rodriguez2016_x = np.linspace(
        max(rodriguez2016_lo[:, 0].min(), rodriguez2016_hi[:, 0].min()),
        min(rodriguez2016_lo[:, 0].max(), rodriguez2016_hi[:, 0].max()),
        5000,
    )
    rodriguez2016_lo_y = np.interp(
        rodriguez2016_x, rodriguez2016_lo[:, 0], rodriguez2016_lo[:, 1]
    )
    rodriguez2016_hi_y = np.interp(
        rodriguez2016_x, rodriguez2016_hi[:, 0], rodriguez2016_hi[:, 1]
    )
    ax.fill_between(
        np.log10(rodriguez2016_x),
        rodriguez2016_lo_y,
        rodriguez2016_hi_y,
        alpha=0.2,
        color="tab:green",
    )

    ax.set_ylabel("$\U0001D453_{ex-situ}$", fontsize=fontsize)
    ax.set_xlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])")

    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        left=True,
        length=6,
        width=1,
        labelsize=labelsize,
    )

    ax.minorticks_on()
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        left=True,
        length=3,
        width=0.8,
        labelsize=labelsize,
    )
    ax.legend(fontsize=legendsize)
    ax.set_title("z = 0", fontsize=titlesize)
    ax.set_xlim(logsm_obs_min, logsm_obs_max)
    ax.set_ylim(0, 0.95)

    fig.savefig(
        savedir + "/" + run_label + "_ex_situ_frac.png",
        dpi=600,
    )

    if plt_show:
        plt.show()
    plt.close()
