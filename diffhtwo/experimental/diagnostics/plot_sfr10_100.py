import matplotlib.pyplot as plt

from diffhtwo.experimental.kernels.lc_phot_kern import multiband_lc_phot_kern
from diffhtwo.experimental.kernels.sfr_tau import get_logsfr_tau

from ..tab_blue_orange_cmap import make_cmap
from .plot_utils import make_thresholded_reduce_C_function, percentile_norm

plt.rc("font", family="serif", serif=["Times New Roman"])
cmap = make_cmap()


def plot_sfr10_100(
    ran_key,
    param_collection,
    z_min,
    z_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    run_label,
    savedir,
    lgmp_min=10.5,
    lgmp_max=15.0,
    num_halos=5000,
    lgsm_cut=8.0,
    gridsize=(100, 60),
    threshold=1e-6,
    percentile=(0.01, 98),
    cmap=cmap,
    plt_show=True,
):
    lc_data, phot_data, gal_weight = multiband_lc_phot_kern(
        ran_key,
        param_collection,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        n_z_phot_table=n_z_phot_table,
    )

    logsfr_100Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.1)
    logsfr_10Myr = get_logsfr_tau(phot_data, lc_data, ssp_data, tau_gyr=0.01)
    lgsfr10_100 = logsfr_10Myr - logsfr_100Myr
    sel = phot_data.logsm_obs > lgsm_cut

    reduce_C_function = make_thresholded_reduce_C_function(
        gal_weight, threshold=threshold
    )

    fig, ax = plt.subplots(
        1,
        figsize=(7.1, 5.5),
        constrained_layout=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    fontsize = 12
    labelsize = 11

    hb = ax.hexbin(
        lc_data.z_obs[sel],
        lgsfr10_100[sel],
        C=gal_weight[sel],
        reduce_C_function=reduce_C_function,
        cmap=cmap,
        gridsize=gridsize,
        extent=(z_min, z_max, -1, 1),
    )
    all_counts = [hb.get_array()]
    norm = percentile_norm(all_counts, lo=percentile[0], hi=percentile[1])
    hb.set_norm(norm)

    ax.axhline(0, c="k", ls=":", alpha=0.5)

    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=False,
        length=6,
        width=1,
        labelsize=labelsize,
    )
    ax.minorticks_on()
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        length=3,
        width=0.8,
        labelsize=labelsize,
    )
    ax.set_xlabel("redshift", fontsize=fontsize)
    ax.set_ylabel("log$_{10}$ (SFR$_{10}$ / SFR$_{100}$)", fontsize=fontsize)

    fig.colorbar(hb, ax=ax, label="$\U0001D453$", pad=0.01)
    fig.savefig(
        savedir + "/" + run_label + "_sfr10_100.png",
        dpi=400,
    )
    if plt_show:
        plt.show()
    plt.close()
