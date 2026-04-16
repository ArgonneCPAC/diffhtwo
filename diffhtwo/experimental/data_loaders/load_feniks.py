from collections import namedtuple

import numpy as np
from astropy.io import ascii

from .. import diffndhist, n_mag
from ..defaults import FENIKS_Z_MAX, FENIKS_Z_MIN
from ..latin_hypercube import latin_hypercube as lh
from ..utils import zbin_volume

FENIKS_PHOT_BASENAME = "feniks_selected.cat"
FENIKS_Z_BASENAME = "feniks_z_selected.ecsv"

FENIKS = namedtuple(
    "FENIKS", ["dataset", "dim_labels"]
)  # , "lh_centroids", "d_centroids", "lg_n_data_err_lh"],


def get_mag_ab(phot_table, col_name, ZP=25):
    mag_ab = -2.5 * np.log10(phot_table[col_name]) + ZP
    mag_ab[~np.isfinite(mag_ab)] = -99.0

    return mag_ab.data


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

    nan_mask = (
        (megacam_uS != -99.0)
        | (hsc_g != -99.0)
        | (hsc_r != -99.0)
        | (hsc_i != -99.0)
        | (nb816 != -99)
        | (hsc_z != -99.0)
        | (nb921 != -99)
        | (video_Y != -99)
        | (uds_J != -99.0)
        | (uds_H != -99.0)
        | (uds_K != -99.0)
        | (uds_Ktot != -99.0)
    )
    print(nan_mask.sum() / nan_mask.size)

    megacam_uS = megacam_uS[nan_mask]
    hsc_g = hsc_g[nan_mask]
    hsc_r = hsc_r[nan_mask]
    hsc_i = hsc_i[nan_mask]
    nb816 = nb816[nan_mask]
    hsc_z = hsc_z[nan_mask]
    nb921 = nb921[nan_mask]
    video_Y = video_Y[nan_mask]
    uds_J = uds_J[nan_mask]
    uds_H = uds_H[nan_mask]
    uds_K = uds_K[nan_mask]
    uds_Ktot = uds_Ktot[nan_mask]

    zout = zout[nan_mask]

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

    return FENIKS(
        dataset,
        dim_labels,
    )
