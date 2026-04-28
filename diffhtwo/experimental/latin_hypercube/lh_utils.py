from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky.mass_functions import mc_hosts

from .. import diffndhist
from ..lc_utils import zbin_volume
from ..lightcone_generators import generate_lc_data
from ..n_mag import get_n_data_err


def get_data_mag_func(dataset, z_min, z_max, data_sky_area_degsq, dmag=0.2):
    dataset_z_sel = (dataset[:, -1] > z_min) & (dataset[:, -1] < z_max)
    mags = dataset[:, -2][dataset_z_sel]

    mag_bin_edges = np.arange(mags.min(), mags.max(), dmag)
    N, _ = np.histogram(mags, bins=mag_bin_edges)

    vol_mpc3 = zbin_volume(data_sky_area_degsq, zlow=z_min, zhigh=z_max).value
    lg_n, lg_n_avg_err = get_n_data_err(N, vol_mpc3)

    return mag_bin_edges, lg_n, lg_n_avg_err


def get_subset_lh_z(
    ran_key,
    dataset,
    z_min,
    z_max,
    data_sky_area_degsq,
    N_centroids,
    ssp_data,
    num_halos=1000,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_z_phot_table=15,
):
    z_sel = (dataset.lh_centroids[:, -1] > z_min) & (
        dataset.lh_centroids[:, -1] < z_max
    )

    print("z N_centroids: " + str(z_sel.sum()))

    lh_centroids_z_subset = dataset.lh_centroids[z_sel]
    d_centroids_z_subset = dataset.d_centroids[z_sel]
    N_data_subset = dataset.N_data[z_sel]

    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        data_sky_area_degsq,
        ssp_data,
        dataset.tcurves,
        z_phot_table,
    )

    lc_data = generate_lc_data(*lc_args)

    dataset = dataset._replace(
        lh_centroids=lh_centroids_z_subset,
        d_centroids=d_centroids_z_subset,
        N_data=N_data_subset,
    )

    DATASET = namedtuple(
        "DATASET",
        dataset._fields + ("lc_data",),
    )
    dataset = DATASET(
        *dataset,
        lc_data,
    )

    dataset = get_subset_lh(dataset, N_centroids)

    # (
    #     norm_band_bin_edges,
    #     norm_band_lg_n,
    #     norm_band_lg_n_avg_err,
    # ) = get_data_mag_func(dataset.dataset, z_min, z_max, data_sky_area_degsq)

    # DATASET = namedtuple(
    #     "DATASET",
    #     dataset._fields
    #     + (
    #         "norm_band_bin_edges",
    #         "norm_band_lg_n",
    #         "norm_band_lg_n_avg_err",
    #     ),
    # )
    # dataset = DATASET(
    #     *dataset,
    #     norm_band_bin_edges,
    #     norm_band_lg_n,
    #     norm_band_lg_n_avg_err,
    # )

    return dataset


def get_subset_lh(dataset, N_centroids):
    lh_idx = np.random.choice(
        dataset.lh_centroids.shape[0], size=N_centroids, replace=False
    )

    lh_centroids_subset = dataset.lh_centroids[lh_idx]
    d_centroids_subset = dataset.d_centroids[lh_idx]
    N_data_subset = dataset.N_data[lh_idx]
    # lg_n_data_err_lh_subset = dataset.lg_n_data_err_lh[:, lh_idx]

    # lh_vol_mpc3_subset = dataset.lc_data.lh_vol_mpc3[lh_idx]
    # lc_data_subset = dataset.lc_data._replace(lh_vol_mpc3=lh_vol_mpc3_subset)

    dataset = dataset._replace(
        lh_centroids=lh_centroids_subset,
        d_centroids=d_centroids_subset,
        N_data=N_data_subset,
        # lg_n_data_err_lh=lg_n_data_err_lh_subset,
        # lc_data=lc_data_subset,
    )

    return dataset


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
