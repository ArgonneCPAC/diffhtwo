import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..kernels.line_dustfree import linelum_gal_dustfree
from ..lightcone_generators import generate_lc_data

plt.rc("font", family="serif", serif=["Times New Roman"])

COLORS_Z = ["#2d0b52", "#0491a1", "#a1d661", "#ee6920", "#8a0f17"]


def plot_halpha_uv_ratio(
    ran_key,
    zbins,
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
    plt_show=True,
):
    Range = [-4, 0]
    bins = 20
    titlesize = 11
    labelsize = 11
    legendsize = 8
    alpha = 0.6

    fig, ax = plt.subplots(1, len(zbins), figsize=(7.1, 2.1), constrained_layout=True)

    for zbin in range(0, len(zbins)):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]
        z_med = str(np.median(zbins[zbin]))
        ax[zbin].set_title(r"$z = $" + z_med, fontsize=titlesize)

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

        dustfree_linelum_gal = linelum_gal_dustfree(
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

        log_halpha_uv = np.log10(
            dustfree_linelum_gal[:, 0] / dustfree_linelum_gal[:, 1]
        )

        ax[zbin].hist(
            log_halpha_uv,
            weights=gal_weight,
            range=Range,
            bins=bins,
            color=COLORS_Z[zbin],
            alpha=alpha,
            density=True,
        )
        ax[zbin].axvspan(-1.93, -1.78, alpha=0.3, color="k", label="Mehta+ 2023")

    ax[-1].legend(fontsize=legendsize)
    fig.supxlabel("log$_{10}$ (L$_{H\u03b1}$ / L$_{UV}$)", fontsize=labelsize)
    fig.supylabel("PDF", fontsize=labelsize)

    fig.savefig(
        savedir + "/" + run_label + "_halpha_uv_ratio" + ".png",
        dpi=400,
    )

    if plt_show:
        plt.show()

    plt.close()
