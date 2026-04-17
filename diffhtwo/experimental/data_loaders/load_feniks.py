from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii

from .. import diffndhist, n_mag
from ..defaults import FENIKS_AREA_DEG2, FENIKS_MAGK_THRESH, FENIKS_Z_MAX, FENIKS_Z_MIN
from ..latin_hypercube import latin_hypercube as lh
from ..utils import zbin_volume

FENIKS_PHOT_BASENAME = "feniks_selected.cat"
FENIKS_Z_BASENAME = "feniks_z_selected.ecsv"

FENIKS = namedtuple(
    "FENIKS",
    [
        "dataset",
        "dim_labels",
        "lh_centroids",
        "d_centroids",
        "lg_n_data_err_lh",
        "lg_n_data_err_lh_old",
    ],
)

SIG = 2.5
N_CENTROIDS = 3000
D_MAG = 0.5
D_Z = 0.5


def get_mag_ab(phot_table, col_name, ZP=25):
    mag_ab = -2.5 * np.log10(phot_table[col_name]) + ZP
    mag_ab[~np.isfinite(mag_ab)] = -99.0

    return mag_ab.data


def modulate_dmag(dataset, lh_centroid, Nmax, dmag, D_MAG_MAX=2.5):
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
    return 1.0, Nbin


def enlarge_lh_bins(dataset, lh_centroids, Nmax, dmag=D_MAG, dz=D_Z):
    N_data_lh = []
    dmag_centroids = []

    for lh_centroid in lh_centroids:
        dmag_centroid, Nbin = modulate_dmag(dataset, lh_centroid, Nmax, dmag)
        dmag_centroids.append(dmag_centroid)
        N_data_lh.append(Nbin)

    dmag_centroids = jnp.array(dmag_centroids)
    dmag_centroids = dmag_centroids.reshape(dmag_centroids.size, 1)

    dmag_centroids = jnp.array(dmag_centroids)
    dmag_centroids = jnp.broadcast_to(dmag_centroids, lh_centroids.shape)

    # set width in redshift dimension to default D_Z
    d_centroids = dmag_centroids.at[:, -1].set(dz)

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)

    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )

    return N_data_lh, d_centroids


def get_feniks_data(drn, phot=FENIKS_PHOT_BASENAME, zout=FENIKS_Z_BASENAME):
    phot = ascii.read(drn + "/" + phot)
    zout = ascii.read(drn + "/" + zout)

    # get mags
    megacam_uS = get_mag_ab(phot, "fcol_MegaCam_uS")
    hsc_g = get_mag_ab(phot, "fcol_HSC_G")
    hsc_r = get_mag_ab(phot, "fcol_HSC_R")
    hsc_i = get_mag_ab(phot, "fcol_HSC_I")
    nb816 = get_mag_ab(phot, "fcol_NB0816")
    hsc_z = get_mag_ab(phot, "fcol_HSC_Z")
    nb921 = get_mag_ab(phot, "fcol_NB0921")
    video_Y = get_mag_ab(phot, "fcol_VIDEO_Y")
    uds_J = get_mag_ab(phot, "fcol_UDS_J")
    uds_H = get_mag_ab(phot, "fcol_UDS_H")
    uds_K = get_mag_ab(phot, "fcol_UDS_K")
    uds_Ktot = get_mag_ab(phot, "ftot_Kuds")

    # mask nans
    nans = (
        (megacam_uS == -99.0)
        | (hsc_g == -99.0)
        | (hsc_r == -99.0)
        | (hsc_i == -99.0)
        | (nb816 == -99)
        | (hsc_z == -99.0)
        | (nb921 == -99)
        | (video_Y == -99)
        | (uds_J == -99.0)
        | (uds_H == -99.0)
        | (uds_K == -99.0)
        | (uds_Ktot == -99.0)
    )

    megacam_uS = megacam_uS[~nans]
    hsc_g = hsc_g[~nans]
    hsc_r = hsc_r[~nans]
    hsc_i = hsc_i[~nans]
    nb816 = nb816[~nans]
    hsc_z = hsc_z[~nans]
    nb921 = nb921[~nans]
    video_Y = video_Y[~nans]
    uds_J = uds_J[~nans]
    uds_H = uds_H[~nans]
    uds_K = uds_K[~nans]
    uds_Ktot = uds_Ktot[~nans]

    zout = zout[~nans]

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_ri = hsc_r - hsc_i
    hsc_i816 = hsc_i - nb816
    hsc_816z = nb816 - hsc_z
    hsc_z921 = hsc_z - nb921
    hsc_video_921Y = nb921 - video_Y
    video_uds_YJ = video_Y - uds_J
    uds_JH = uds_J - uds_H
    uds_HK = uds_H - uds_K

    # stack dataset
    dataset = np.vstack(
        (
            megacam_hsc_uSg,
            hsc_gr,
            hsc_ri,
            hsc_i816,
            hsc_816z,
            hsc_z921,
            hsc_video_921Y,
            video_uds_YJ,
            uds_JH,
            uds_HK,
            megacam_uS,
            uds_Ktot,
            zout["z_phot"],
        )
    ).T

    # mask redshift
    z_mask = (zout["z_phot"] > FENIKS_Z_MIN) & (zout["z_phot"] <= FENIKS_Z_MAX)
    dataset = dataset[z_mask]
    zout = zout[z_mask]

    dim_labels = [
        "u - g",
        "g - r",
        "r - i",
        "i - nb816",
        "nb816 - z",
        "z - nb921",
        "nb921 - Y",
        "Y - J",
        "J - H",
        "H - K",
        "u",
        "K",
        "redshift",
    ]

    # get number densities in latin hypercube bins
    mu = np.mean(dataset, axis=0)
    mu[-1] = mu[-1] + 0.2
    mu[-2] = mu[-2]
    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(mu, cov, SIG, N_CENTROIDS, seed=None)

    redshift_mask = (lh_centroids[:, -1] > FENIKS_Z_MIN) & (
        lh_centroids[:, -1] < FENIKS_Z_MAX
    )
    k_mask = lh_centroids[:, -2] < FENIKS_MAGK_THRESH
    lh_centroids = lh_centroids[redshift_mask & k_mask]

    # run initial diffndhist
    d_centroids = jnp.ones_like(lh_centroids) * D_MAG
    d_centroids = d_centroids.at[:, -1].set(D_Z)
    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    N_data_lh_old = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )
    Nmax = N_data_lh_old.max()
    vol_mpc3 = zbin_volume(
        FENIKS_AREA_DEG2, zlow=FENIKS_Z_MIN, zhigh=FENIKS_Z_MAX
    ).value

    lg_n_old, lg_n_avg_err_old = n_mag.get_n_data_err(N_data_lh_old, vol_mpc3)
    lg_n_data_err_lh_old = jnp.vstack((lg_n_old, lg_n_avg_err_old))

    # run final diffndhist
    N_data_lh, d_centroids = enlarge_lh_bins(dataset, lh_centroids, Nmax)

    lg_n, lg_n_avg_err = n_mag.get_n_data_err(N_data_lh, vol_mpc3)
    lg_n_data_err_lh = jnp.vstack((lg_n, lg_n_avg_err))

    return FENIKS(
        dataset,
        dim_labels,
        lh_centroids,
        d_centroids,
        lg_n_data_err_lh,
        lg_n_data_err_lh_old,
    )
