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

    z_mask = (zout["z_phot"] > FENIKS_Z_MIN) & (zout["z_phot"] <= FENIKS_Z_MAX)

    dataset = np.vstack(
        (
            megacam_hsc_uSg[z_mask],
            hsc_gr[z_mask],
            hsc_ri[z_mask],
            hsc_i816[z_mask],
            hsc_816z[z_mask],
            hsc_z921[z_mask],
            hsc_video_921Y[z_mask],
            video_uds_YJ[z_mask],
            uds_JH[z_mask],
            uds_HK[z_mask],
            megacam_uS[z_mask],
            uds_Ktot[z_mask],
            zout["z_phot"][z_mask],
        )
    ).T
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
