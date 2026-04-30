from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from diffsky.mass_functions import mc_hosts
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders.defaults import TransmissionCurve

from .. import diffndhist
from ..defaults import (
    DATASET,
    FENIKS_AREA_DEG2,
    FENIKS_MAGK_THRESH,
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
)
from ..latin_hypercube import latin_hypercube as lh
from ..utils import (
    get_feniks_filter_number_from_translate_file,
    get_tcurve,
)

FENIKS = namedtuple("FENIKS", DATASET._fields)

PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"
TRANSLATE = "filters_w_FENIKS.translate"
FILTER_INFO = "kz_FILTER.RES.latest.info"
TCURVES_FILE = "kz_FILTER.RES.latest"

LH_SIG = 3
LH_N_CENTROIDS = 50_000

D_MAG = 0.7
D_Z = 0.5


def get_mag_ab(phot_table, col_name, ZP=25):
    mag_ab = -2.5 * np.log10(phot_table[col_name]) + ZP
    mag_ab[~np.isfinite(mag_ab)] = -99.0

    return mag_ab.data


def get_lh_centroids(
    dataset,
    z_min,
    z_max,
    mag_thresh,
    lh_n_centroids=LH_N_CENTROIDS,
    lh_sig=LH_SIG,
    d_mag=D_MAG,
    d_z=D_Z,
):
    mu = np.mean(dataset, axis=0)
    mu[0] = mu[0] + 1  # u - g
    mu[1] = mu[1] + 0.5  # g - r
    mu[2] = mu[2] - 0.1  # r - i
    mu[-2] = mu[-2] - 1.8  # K
    mu[-1] = mu[-1] + 0.2  # redshift
    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, lh_sig, lh_n_centroids, seed=None
    )

    redshift_mask = (lh_centroids[:, -1] > (z_min + (d_z / 2))) & (
        lh_centroids[:, -1] < (z_max - (d_z / 2))
    )
    k_mask = lh_centroids[:, -2] <= mag_thresh
    lh_centroids = lh_centroids[redshift_mask & k_mask]

    redshift_centers = [0.45, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95]
    k_mins = [16, 17.8, 19, 19.7, 20.2, 20.8, 21.2, 21.8]
    coeffs = np.polyfit(redshift_centers, k_mins, deg=2)
    k_min = np.poly1d(coeffs)
    k_complete = lh_centroids[:, -2] > k_min(lh_centroids[:, -1])
    lh_centroids = lh_centroids[k_complete]

    d_centroids = jnp.ones_like(lh_centroids) * d_mag
    d_centroids = d_centroids.at[:, -1].set(d_z)

    return lh_centroids, d_centroids


def get_feniks_data(
    drn,
    ran_key,
    ssp_data,
    lh_sig=LH_SIG,
    lh_n_centroids=LH_N_CENTROIDS,
    z_min=FENIKS_Z_MIN,
    z_max=FENIKS_Z_MAX,
    mag_thresh=FENIKS_MAGK_THRESH,
    data_sky_area_degsq=FENIKS_AREA_DEG2,
    num_halos=1000,
    lc_sky_area_degsq=FENIKS_AREA_DEG2,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    n_z_phot_table=120,
    phot=PHOT,
    zout=ZOUT,
    translate=TRANSLATE,
    filter_info=FILTER_INFO,
    tcurves_file=TCURVES_FILE,
    dmag=D_MAG,
    dz=D_Z,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    # Transmission curves
    tcurves = []
    feniks_filters = [
        "MegaCam_uS",
        "HSC_G",
        "HSC_R",
        "HSC_I",
        "HSC_Z",
        "VIDEO_Y",
        "UDS_J",
        "UDS_H",
        "UDS_K",  # mag_column, mag_thresh_column
    ]
    mag_columns = [8]
    mag_thresh_column = 8

    translate = ascii.read(drn + "/" + translate, header_start=None)
    filter_info = drn + "/" + filter_info
    tcurves_file = drn + "/" + tcurves_file

    for feniks_filter in feniks_filters:
        feniks_filter_number = get_feniks_filter_number_from_translate_file(
            translate, feniks_filter
        )
        feniks_filter_wave_aa, feniks_filter_trans = get_tcurve(
            feniks_filter_number, filter_info, tcurves_file
        )
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    phot = ascii.read(drn + "/" + phot)
    zout = ascii.read(drn + "/" + zout)
    N_obj_pre_cuts = len(zout)

    clean = (
        (phot["fcol_MegaCam_uS"] != -99)
        & (phot["fcol_HSC_G"] != -99)
        & (phot["fcol_HSC_R"] != -99)
        & (phot["fcol_HSC_I"] != -99)
        & (phot["fcol_HSC_Z"] != -99)
        & (phot["fcol_VIDEO_Y"] != -99)
        & (phot["fcol_UDS_J"] != -99)
        & (phot["fcol_UDS_H"] != -99)
        & (phot["fcol_UDS_K"] != -99)
    )

    phot = phot[clean]
    zout = zout[clean]

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
    uds_Ktot = get_mag_ab(phot, "ftot_Kuds")

    # mask nans
    nans = (
        (megacam_uS == -99.0)
        | (hsc_g == -99.0)
        | (hsc_r == -99.0)
        | (hsc_i == -99.0)
        | (hsc_z == -99.0)
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
    hsc_z = hsc_z[~nans]
    video_Y = video_Y[~nans]
    uds_J = uds_J[~nans]
    uds_H = uds_H[~nans]
    uds_K = uds_K[~nans]
    uds_Ktot = uds_Ktot[~nans]

    zout = zout[~nans]
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
            uds_Ktot,
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
            uds_Ktot,
            zout["z_phot"],
        )
    ).T

    # mask redshift
    z_mask = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)
    dataset = dataset[z_mask]
    mags = mags[z_mask]
    zout = zout[z_mask]

    lh_centroids, d_centroids = get_lh_centroids(
        dataset,
        z_min,
        z_max,
        mag_thresh,
    )

    # run initial diffndhist with fixed dmag
    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)
    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    return FENIKS(
        dataset,
        mags,
        tcurves,
        mag_columns,
        mag_thresh_column,
        mag_thresh,
        frac_cat,
        lh_centroids,
        d_centroids,
        N_data_lh,
        data_sky_area_degsq,
        # lg_n_data_err_lh,
        # lc_data,
    )
