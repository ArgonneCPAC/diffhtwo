import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from diffhtwo.experimental.kernels.lc_photline_kern import lc_photline_kern

from ..kernels.sfr_tau import get_logsfr_tau

plt.rc("font", family="serif", serif=["Times New Roman"])


def plot_halpha_mstar(
    ran_key,
    param_collection,
    zbins,
    ssp_data_ha,
    tcurves,
    halpha_wave_aa,
    num_halos,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_max=15.0,
    n_z_phot_table=2,
    mc_merge=1,
    xlim=(8, 12),
    ylim=(38, 44),
    gridsize=(70, 50),
    vmin=-12,
    vmax=-8,
    plt_show=True,
):
    line_wave_table = jnp.array([halpha_wave_aa])
    n_z_bins = len(zbins)

    phot_data, phot_randoms, photline_data, lc_data = lc_photline_kern(
        ran_key,
        param_collection,
        zbins[0][0],
        zbins[-1][-1],
        num_halos,
        ssp_data_ha,
        tcurves,
        line_wave_table,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        n_z_phot_table=n_z_phot_table,
        mc_merge=mc_merge,
    )
    gal_weight = lc_data.cen_weight * lc_data.sat_weight

    lg_L_halpha = np.log10(photline_data.linelum_gal[:, 0])
    lgsm_obs = phot_data.logsm_obs

    lgsfr = get_logsfr_tau(phot_data, lc_data, ssp_data_ha, tau_gyr=0.01)
    lgssfr = lgsfr - lgsm_obs

    unmerged = phot_data.p_merge < 0.5

    fig, ax = plt.subplots(
        1,
        n_z_bins,
        figsize=(7.1, 2.3),
        constrained_layout=True,
    )
    labelsize = 9
    fontsize = 10

    for z in range(n_z_bins):
        z_min, z_max = zbins[z]
        z_label = np.median(zbins[z])

        z_sel = (lc_data.z_obs > z_min) & (lc_data.z_obs < z_max)

        lg_L_halpha_selected = lg_L_halpha[unmerged & z_sel]
        lgsm_obs_selected = lgsm_obs[unmerged & z_sel]
        lgssfr_selected = lgssfr[unmerged & z_sel]
        gal_weight_selected = gal_weight[unmerged & z_sel]

        idx = np.arange(len(lgssfr_selected))

        def reduce_C_function(i):
            i = np.asarray(i, dtype=int)
            return np.log10(
                np.average(10 ** lgssfr_selected[i], weights=gal_weight_selected[i])
            )

            # ax[z].scatter(
            #     lgsm_obs, lg_L_halpha, s=1, alpha=0.5, c=logsfr_100Myr[unmerged & z_sel]
            # )

        hb = ax[z].hexbin(
            lgsm_obs_selected,
            lg_L_halpha_selected,
            C=idx,
            reduce_C_function=reduce_C_function,
            cmap="coolwarm_r",
            gridsize=gridsize,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            vmin=vmin,
            vmax=vmax,
        )
        ax[z].set_title(f"z={z_label}")

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
    fig.supxlabel(r"log$_{10}$ (M$_{*}$ [M$_{\odot}$])", fontsize=fontsize)
    fig.supylabel("log$_{10}$ (L$_{H\u03b1}$ [erg/s])", fontsize=fontsize)
    fig.colorbar(hb, ax=ax, label=r"log$_{10}$ (sSFR$_{10\,\mathrm{Myr}}$ [yr$^{-1}$])")

    fig.savefig(savedir + "/" + run_label + "_halpha_mstar.png", dpi=600)

    if plt_show:
        plt.show()
    plt.close()

    plt.show()
