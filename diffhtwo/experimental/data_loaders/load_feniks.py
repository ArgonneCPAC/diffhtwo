import warnings
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from diffsky import diffndhist_lomem
from dsps.data_loaders.defaults import TransmissionCurve

from ..defaults import (
    FENIKS_AREA_DEG2,
    FENIKS_MAGK_THRESH,
    FENIKS_MAGOTHER_THRESH,
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
    Dataset,
    FilterInfo,
)
from ..latin_hypercube import latin_hypercube as lh
from ..utils import load_feniks_tcurve

BASE_PATH = Path(__file__).resolve().parent.parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"

Feniks = namedtuple("Feniks", Dataset._fields)

LH_SIG = 3.0
LH_N_CENTROIDS = 15_000

# LH_D_MAG = 0.5
LH_D_Z = 0.5


def get_mag_ab(phot_table, col_name, ZP=25):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_ab = -2.5 * np.log10(phot_table[col_name]) + ZP

    mag_ab[~np.isfinite(mag_ab)] = -99.0

    return mag_ab.data


def refresh_lh_centroids(DATASET, lh_d_mag):
    lh_centroids, d_centroids = get_lh_centroids(DATASET.dataset, lh_d_mag)

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)
    N_data_lh = diffndhist_lomem.tw_ndhist(
        DATASET.dataset,
        dataset_sig,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    DATASET = DATASET._replace(
        lh_centroids=lh_centroids, d_centroids=d_centroids, N_data=N_data_lh
    )

    return DATASET


def get_lh_centroids(dataset, lh_d_mag):
    mu = np.mean(dataset, axis=0)

    # mu[0] = mu[0] + 0.4  # u - g
    # mu[1] = mu[1] + 0.0  # g - r
    # mu[2] = mu[2] + 0.0  # r - i
    # mu[3] = mu[3] + 0.1  # z - Y
    # mu[4] = mu[4] + 0.15  # z - Y
    # mu[5] = mu[5] + 0.0  # Y - J
    # mu[6] = mu[6] + 0.0  # J - H
    # mu[8] = mu[8] + 0.0  # u

    # mu[-2] = mu[-2] - 1  # K
    # mu[-1] = mu[-1] + 0.5  # redshift

    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, LH_SIG, LH_N_CENTROIDS, seed=None
    )

    redshift_mask = (lh_centroids[:, -1] > (FENIKS_Z_MIN + (LH_D_Z / 2))) & (
        lh_centroids[:, -1] < (FENIKS_Z_MAX - (LH_D_Z / 2))
    )
    k_mask = lh_centroids[:, -2] < FENIKS_MAGK_THRESH
    u_mask = lh_centroids[:, -3] < FENIKS_MAGOTHER_THRESH
    lh_centroids = lh_centroids[redshift_mask & k_mask & u_mask]

    redshift_centers = [0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95]
    k_mins = [16, 17.8, 19, 19.7, 20.2, 20.8, 21.2, 21.8]
    coeffs = np.polyfit(redshift_centers, k_mins, deg=2)
    k_min = np.poly1d(coeffs)
    k_bright = lh_centroids[:, -2] > k_min(lh_centroids[:, -1])
    lh_centroids = lh_centroids[k_bright]

    d_centroids = jnp.ones_like(lh_centroids) * lh_d_mag
    d_centroids = d_centroids.at[:, -1].set(LH_D_Z)

    return lh_centroids, d_centroids


def get_feniks_data(
    drn,
    ran_key,
    ssp_data,
    lh_d_mag,
    phot=PHOT,
    zout=ZOUT,
):
    # Transmission curves and filter mag thresholds

    feniks_mag_thresh = FeniksFilters(
        MegaCam_uS=FENIKS_MAGOTHER_THRESH,
        HSC_G=FENIKS_MAGOTHER_THRESH,
        HSC_R=FENIKS_MAGOTHER_THRESH,
        HSC_I=FENIKS_MAGOTHER_THRESH,
        HSC_Z=FENIKS_MAGOTHER_THRESH,
        VIDEO_Y=FENIKS_MAGOTHER_THRESH,
        UDS_J=FENIKS_MAGOTHER_THRESH,
        UDS_H=FENIKS_MAGOTHER_THRESH,
        UDS_K=FENIKS_MAGK_THRESH,
    )
    feniks_in_lh = FeniksFilters(
        MegaCam_uS=True,
        HSC_G=False,
        HSC_R=False,
        HSC_I=False,
        HSC_Z=False,
        VIDEO_Y=False,
        UDS_J=False,
        UDS_H=False,
        UDS_K=True,
    )
    tcurves = []
    for feniks_filter in FeniksFilters._fields:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    filter_info = FilterInfo(feniks_mag_thresh, feniks_in_lh, tcurves)

    drn_path = Path(drn)
    phot = ascii.read(drn_path / phot)
    zout = ascii.read(drn_path / zout)

    # get mags
    megacam_uS = get_mag_ab(phot, "fcol_MegaCam_uS")
    hsc_g = get_mag_ab(phot, "fcol_HSC_G")
    hsc_r = get_mag_ab(phot, "fcol_HSC_R")
    hsc_i = get_mag_ab(phot, "fcol_HSC_I")
    hsc_z = get_mag_ab(phot, "fcol_HSC_Z")
    video_Y = get_mag_ab(phot, "fcol_VIDEO_Y")
    uds_J = get_mag_ab(phot, "fcol_UDS_J")
    uds_H = get_mag_ab(phot, "fcol_UDS_H")
    uds_K = get_mag_ab(phot, "fcol_UDS_K")

    # get mag thresh cuts
    mag_thresh = (
        (megacam_uS < feniks_mag_thresh.MegaCam_uS)
        & (hsc_g < feniks_mag_thresh.HSC_G)
        & (hsc_r < feniks_mag_thresh.HSC_R)
        & (hsc_i < feniks_mag_thresh.HSC_I)
        & (hsc_z < feniks_mag_thresh.HSC_Z)
        & (video_Y < feniks_mag_thresh.VIDEO_Y)
        & (uds_J < feniks_mag_thresh.UDS_J)
        & (uds_H < feniks_mag_thresh.UDS_H)
        & (uds_K < feniks_mag_thresh.UDS_K)
    )

    # apply mag_thresh cuts and record n_gals.
    # This is the starting point from which any further cuts will
    # lead to frac_cat (fraction of catalog thrown due to bad data) being calculated
    phot = phot[mag_thresh]
    zout = zout[mag_thresh]
    megacam_uS = megacam_uS[mag_thresh]
    hsc_g = hsc_g[mag_thresh]
    hsc_r = hsc_r[mag_thresh]
    hsc_i = hsc_i[mag_thresh]
    hsc_z = hsc_z[mag_thresh]
    video_Y = video_Y[mag_thresh]
    uds_J = uds_J[mag_thresh]
    uds_H = uds_H[mag_thresh]
    uds_K = uds_K[mag_thresh]

    N_obj_pre_cuts = len(zout)

    # remove mags with bad data in any of the bands
    clean = (
        (megacam_uS != -99)
        & (hsc_g != -99)
        & (hsc_r != -99)
        & (hsc_i != -99)
        & (hsc_z != -99)
        & (video_Y != -99)
        & (uds_J != -99)
        & (uds_H != -99)
        & (uds_K != -99)
    )

    phot = phot[clean]
    zout = zout[clean]
    megacam_uS = megacam_uS[clean]
    hsc_g = hsc_g[clean]
    hsc_r = hsc_r[clean]
    hsc_i = hsc_i[clean]
    hsc_z = hsc_z[clean]
    video_Y = video_Y[clean]
    uds_J = uds_J[clean]
    uds_H = uds_H[clean]
    uds_K = uds_K[clean]

    # mask nans
    # nans = (
    #     (megacam_uS == -99.0)
    #     | (hsc_g == -99.0)
    #     | (hsc_r == -99.0)
    #     | (hsc_i == -99.0)
    #     | (hsc_z == -99.0)
    #     | (video_Y == -99)
    #     | (uds_J == -99.0)
    #     | (uds_H == -99.0)
    #     | (uds_K == -99.0)
    # )

    # megacam_uS = megacam_uS[~nans]
    # hsc_g = hsc_g[~nans]
    # hsc_r = hsc_r[~nans]
    # hsc_i = hsc_i[~nans]
    # hsc_z = hsc_z[~nans]
    # video_Y = video_Y[~nans]
    # uds_J = uds_J[~nans]
    # uds_H = uds_H[~nans]
    # uds_K = uds_K[~nans]

    # zout = zout[~nans]

    N_obj_post_cuts = len(zout)
    frac_cat = N_obj_post_cuts / N_obj_pre_cuts

    mags = np.vstack(
        (
            megacam_uS,
            hsc_g,
            hsc_r,
            hsc_i,
            hsc_z,
            video_Y,
            uds_J,
            uds_H,
            uds_K,
            zout["z_phot"],
        )
    ).T

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_ri = hsc_r - hsc_i
    hsc_iz = hsc_i - hsc_z
    hsc_video_zY = hsc_z - video_Y
    video_uds_YJ = video_Y - uds_J
    uds_JH = uds_J - uds_H
    uds_HK = uds_H - uds_K

    # stack colors_mag
    dataset = np.vstack(
        (
            megacam_hsc_uSg,
            hsc_gr,
            hsc_ri,
            hsc_iz,
            hsc_video_zY,
            video_uds_YJ,
            uds_JH,
            uds_HK,
            megacam_uS,
            uds_K,
            zout["z_phot"],
        )
    ).T

    dataset_dim_labels = [
        r"$uS_{MegaCam} - g_{HSC}$",
        r"$g_{HSC} - r_{HSC}$",
        r"$r_{HSC} - i_{HSC}$",
        r"$i_{HSC} - z_{HSC}$",
        r"$z_{HSC} - Y_{VIDEO}$",
        r"$Y_{VIDEO} - J_{UDS}$",
        r"$J_{UDS} - H_{UDS}$",
        r"$H_{UDS} - K_{UDS}$",
        r"$uS_{MegaCam}$",
        r"$K_{UDS}$",
        r"$redshift$",
    ]

    mags_labels = [
        r"$uS_{MegaCam}$",
        r"$g_{HSC}$",
        r"$r_{HSC}$",
        r"$i_{HSC}$",
        r"$z_{HSC}$",
        r"$Y_{VIDEO}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
    ]

    # mask redshift
    z_mask = (zout["z_phot"] > FENIKS_Z_MIN) & (zout["z_phot"] <= FENIKS_Z_MAX)
    dataset = dataset[z_mask]
    mags = mags[z_mask]
    zout = zout[z_mask]

    lh_centroids, d_centroids = get_lh_centroids(dataset, lh_d_mag)

    # run initial diffndhist_lomem with fixed dmag
    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)

    N_data_lh = diffndhist_lomem.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    return Feniks(
        dataset,
        dataset_dim_labels,
        mags,
        mags_labels,
        filter_info,
        frac_cat,
        lh_centroids,
        d_centroids,
        N_data_lh,
        lh_d_mag,
        LH_D_Z,
        FENIKS_AREA_DEG2,
    )


FeniksFilters = namedtuple(
    "FeniksFilters",
    [
        "MegaCam_uS",
        "HSC_G",
        "HSC_R",
        "HSC_I",
        "HSC_Z",
        "VIDEO_Y",
        "UDS_J",
        "UDS_H",
        "UDS_K",
    ],
)
