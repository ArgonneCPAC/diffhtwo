from collections import namedtuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import diffndhist
from .. import param_utils as pu
from ..lightcone_generators import generate_lc_data


def get_zbins_lh_lc(
    ran_key,
    dataset,
    z_min,
    z_max,
    ssp_data,
    N_centroids,
    lh_N_z_savedir=None,
    num_halos=1000,
    lc_sky_area_degsq=1000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    n_z_phot_table=15,
):
    in_lh = jnp.array(list(dataset.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]
    meta_data = MetaData(
        dataset.filter_info.mag_thresh,
        in_lh_idx,
        dataset.frac_cat,
        dataset.data_sky_area_degsq,
    )

    N_data = []
    lh_centroids = []
    d_centroids = []
    lc_data = []
    for zbin in range(0, len(z_min)):
        z_sel = (dataset.lh_centroids[:, -1] > (z_min[zbin] + (dataset.lh_dz / 2))) & (
            dataset.lh_centroids[:, -1] < (z_max[zbin] - (dataset.lh_dz / 2))
        )

        print(
            f"{z_sel.sum()} centroids available in this z-bin out of which {N_centroids} will be selected"
        )

        lh_centroids_z = dataset.lh_centroids[z_sel]
        d_centroids_z = dataset.d_centroids[z_sel]
        N_data_z = dataset.N_data[z_sel]

        # select first N_centroids with N_data in descending order
        lh_idx = jnp.argsort(N_data_z)[::-1][:N_centroids]

        lh_centroids_z_subset = lh_centroids_z[lh_idx]
        d_centroids_z_subset = d_centroids_z[lh_idx]
        N_data_z_subset = N_data_z[lh_idx]

        plot_N_z_subset(
            N_data_z_subset, N_data_z, z_min[zbin], z_max[zbin], lh_N_z_savedir
        )

        z_phot_table = 10 ** jnp.linspace(
            jnp.log10(z_min[zbin]), jnp.log10(z_max[zbin]), n_z_phot_table
        )
        lc_args = (
            ran_key,
            num_halos,
            z_min[zbin],
            z_max[zbin],
            lgmp_min,
            lgmp_max,
            lc_sky_area_degsq,
            ssp_data,
            dataset.filter_info.tcurves,
            z_phot_table,
        )

        lc_data_z = generate_lc_data(*lc_args)

        N_data.append(N_data_z_subset)
        lh_centroids.append(lh_centroids_z_subset)
        d_centroids.append(d_centroids_z_subset)
        lc_data.append(lc_data_z)

    # prepare for vmapping over these fields during optimization
    N_data = jnp.array(N_data)
    lh_centroids = jnp.array(lh_centroids)
    d_centroids = jnp.array(d_centroids)
    lc_data = pu.stack_lc_data(lc_data)

    fitting_data = FittingData(N_data, lh_centroids, d_centroids, lc_data)

    return meta_data, fitting_data


def get_single_zbin_lh_lc(
    ran_key,
    dataset,
    z_min,
    z_max,
    ssp_data,
    N_centroids,
    lh_N_z_savedir=None,
    num_halos=1000,
    lc_sky_area_degsq=1000,
    lgmp_min=10.0,
    lgmp_max=15.0,
    n_z_phot_table=15,
):
    in_lh = jnp.array(list(dataset.filter_info.in_lh._asdict().values()))
    in_lh_idx = jnp.where(in_lh)[0]

    meta_data = MetaData(
        dataset.filter_info.mag_thresh,
        in_lh_idx,
        dataset.frac_cat,
        dataset.data_sky_area_degsq,
    )

    z_sel = (dataset.lh_centroids[:, -1] > (z_min + (dataset.lh_dz / 2))) & (
        dataset.lh_centroids[:, -1] < (z_max - (dataset.lh_dz / 2))
    )

    print(
        f"{z_sel.sum()} centroids available in this z-bin out of which {N_centroids} will be selected"
    )

    lh_centroids_z = dataset.lh_centroids[z_sel]
    d_centroids_z = dataset.d_centroids[z_sel]
    N_data_z = dataset.N_data[z_sel]

    # select first N_centroids with N_data in descending order
    lh_idx = jnp.argsort(N_data_z)[::-1][:N_centroids]

    lh_centroids_z_subset = lh_centroids_z[lh_idx]
    d_centroids_z_subset = d_centroids_z[lh_idx]
    N_data_z_subset = N_data_z[lh_idx]

    plot_N_z_subset(N_data_z_subset, N_data_z, z_min, z_max, lh_N_z_savedir)

    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )
    lc_args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        dataset.filter_info.tcurves,
        z_phot_table,
    )

    lc_data_z = generate_lc_data(*lc_args)

    fitting_data = FittingData(
        N_data_z_subset, lh_centroids_z_subset, d_centroids_z_subset, lc_data_z
    )

    return meta_data, fitting_data


def plot_N_z_subset(N_data_z_subset, N_data_z, z_min, z_max, savedir):
    fig, ax = plt.subplots()

    bins = np.linspace(N_data_z.min(), N_data_z.max(), 20)
    label = "N$_{bins, z}$ = " + str(len(N_data_z))
    ax.hist(N_data_z, bins=bins, alpha=0.8, histtype="step", color="k", label=label)

    label = "N$_{bins, sel}$ = " + str(len(N_data_z_subset))
    ax.hist(N_data_z_subset, bins=bins, alpha=0.5, color="k", label=label)

    z_min_label = str(np.round(z_min, 2))
    z_max_label = str(np.round(z_max, 2))
    ax.set_title(z_min_label + " < z < " + z_max_label)
    ax.set_yscale("log")
    ax.set_ylabel("#")
    ax.set_xlabel("counts")
    ax.legend()
    if savedir is not None:
        fig.savefig(savedir + "/lh_N_z" + z_min_label + "-" + z_max_label + ".png")
    plt.close()


def modulate_dmag(dataset, lh_centroid, Nmax, dmag, D_MAG_MAX=1.0):
    lh_centroid = lh_centroid.reshape(1, lh_centroid.size)

    while dmag < D_MAG_MAX:
        sig = jnp.zeros(lh_centroid.shape) + (dmag / 2)
        Nbin = diffndhist.tw_ndhist(
            dataset,
            sig,  # (nbins, ndim)
            lh_centroid - (dmag / 2),  # (nbins, ndim)
            lh_centroid + (dmag / 2),
        )
        if np.isclose(Nmax, Nbin, atol=0.5):
            return dmag, Nbin
        else:
            dmag += 0.1
    return dmag - 0.1, Nbin


def enlarge_lh_bins(dataset, lh_centroids, Nmax, dmag, dz):
    N_data_lh = []
    dmag_centroids = []

    for lh_centroid in lh_centroids:
        dmag_centroid, Nbin = modulate_dmag(dataset, lh_centroid, Nmax, dmag)
        dmag_centroids.append(dmag_centroid)
        N_data_lh.append(Nbin)

    dmag_centroids = jnp.array(dmag_centroids)
    d_centroids = dmag_centroids.reshape(dmag_centroids.size, 1)
    d_centroids = jnp.broadcast_to(d_centroids, lh_centroids.shape)
    d_centroids = d_centroids.at[:, -1].set(dz)

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)

    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )

    return N_data_lh, d_centroids


MetaData = namedtuple(
    "MetaData",
    [
        "mag_thresh",
        "in_lh_idx",
        "frac_cat",
        "data_sky_area_degsq",
    ],
)

FittingData = namedtuple(
    "FittingData", ["N_data", "lh_centroids", "d_centroids", "lc_data"]
)
