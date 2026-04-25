import matplotlib.pyplot as plt
import numpy as np
from diffsky.mass_functions import mc_hosts

from diffhtwo.experimental.lightcone_generators import lc_photdata
from diffhtwo.experimental.n_specphot import get_mc_colors_mags


def N_frac(C_bin, N_tot):
    return len(C_bin) / N_tot


def get_reduce_C_function(x, y, extent):
    within_extent = (
        (x > extent[0]) & (x < extent[1]) & (y > extent[2]) & (y < extent[3])
    )
    x = x[within_extent]
    y = y[within_extent]

    N_tot = len(x)

    def reduce_C_function(C_bin):
        return N_frac(C_bin, N_tot)

    return x, y, np.ones_like(x), reduce_C_function


def compare_colors_mag_z(
    ran_key,
    param_collection,
    z,
    dataset,
    dim_labels,
    data_label,
    savedir,
    dz=0.5,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    sky_area_degsq=0.2,
    n_z_phot_table=10,
    mincnt=3,
    ylim=(-1, 3),
    vmin=5e-3,
    vmax=2e-2,
    cmap="viridis",
):
    z_min = z - (dz / 2)
    z_max = z + (dz / 2)

    z_phot_table = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)

    lc_data = lc_photdata(
        ran_key,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        dataset.lc_data.ssp_data,
        dataset.tcurves,
        z_phot_table,
    )

    obs_color_mag = get_mc_colors_mags(
        ran_key,
        param_collection,
        lc_data,
        dataset.mag_columns,
        dataset.mag_thresh_column,
        dataset.mag_thresh,
    )

    ndims = dataset.dataset.shape[1]

    gridsize = [(60, 30)] * (ndims - 2)
    gridsize.append((50, 20))
    extent = [(z_min, z_max, -2, 4)] * (ndims - 2)
    extent.append((z_min, z_max, 15, dataset.mag_thresh))

    fig, ax = plt.subplots(ndims - 1, 2, figsize=(8.5, 10), width_ratios={0.8, 1})
    fig.subplots_adjust(
        left=0.09, hspace=0.1, bottom=0.06, top=0.96, right=0.9, wspace=0.0
    )

    for i in range(ndims - 1):
        x, y, C, reduce_C_function = get_reduce_C_function(
            obs_color_mag[:, -1], obs_color_mag[:, i], extent[i]
        )
        hb = ax[i, 1].hexbin(
            x,
            y,
            cmap=cmap,
            gridsize=gridsize[i],
            extent=extent[i],
            C=C,
            reduce_C_function=reduce_C_function,
            bins="log",
            # vmin=vmin,
            # vmax=vmax,
            mincnt=mincnt,
            rasterized=True,
        )
        cb = fig.colorbar(hb, ax=ax[i, 1])
        cb.set_label(r"$N_{bin} \;/\; N_{tot}$", fontsize=8)

        vmin = hb.get_array().data.min()
        vmax = hb.get_array().data.max()

        x, y, C, reduce_C_function = get_reduce_C_function(
            dataset.dataset[:, -1], dataset.dataset[:, i], extent[i]
        )

        hb = ax[i, 0].hexbin(
            x,
            y,
            gridsize=gridsize[i],
            extent=extent[i],
            C=C,
            reduce_C_function=reduce_C_function,
            cmap=cmap,
            bins="log",
            vmin=vmin,
            vmax=vmax,
            mincnt=mincnt,
            rasterized=True,
        )
        # cb = fig.colorbar(hb, ax=ax[i, 0])
        # cb.set_label(r"$N_{bin} \;/\; N_{tot}$", fontsize=8)

        ax[i, 0].set_ylabel(dim_labels[i])
        if i != ndims - 2:
            ax[i, 0].set_ylim(ylim)
            ax[i, 1].set_ylim(ylim)
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

        ax[i, 0].set_xlim(z_min - 0.02, z_max + 0.02)
        ax[i, 1].set_xlim(z_min - 0.02, z_max + 0.02)

    ax[ndims - 2, 0].set_xlabel("redshift")
    ax[ndims - 2, 1].set_xlabel("redshift")

    ax[0, 0].set_title(data_label)
    ax[0, 1].set_title("diffsky")

    plt.savefig(
        savedir + "/" + data_label + "_N_frac_z" + str(z) + ".pdf",
        dpi=200,
    )
    plt.show()
