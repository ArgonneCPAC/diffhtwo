from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from diffsky.mass_functions import mc_hosts
from dsps.data_loaders.defaults import TransmissionCurve

from .. import diffndhist, n_mag
from ..defaults import (
    DATASET,
    FENIKS_AREA_DEG2,
    FENIKS_MAGK_THRESH,
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
)
from ..latin_hypercube import latin_hypercube as lh
from ..utils import (
    generate_lc_data,
    get_feniks_filter_number_from_translate_file,
    get_tcurve,
    zbin_volume,
)

FENIKS = namedtuple("FENIKS", DATASET._fields)

PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"
TRANSLATE = "filters_w_FENIKS.translate"
FILTER_INFO = "kz_FILTER.RES.latest.info"
TCURVES_FILE = "kz_FILTER.RES.latest"

LH_SIG = 3
LH_N_CENTROIDS = 3000
D_MAG = 0.5
D_Z = 0.5

FENIKS_N_FLOOR = 0.5


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

    redshift_mask = (lh_centroids[:, -1] > z_min) & (lh_centroids[:, -1] < z_max)
    k_mask = lh_centroids[:, -2] < mag_thresh
    lh_centroids = lh_centroids[redshift_mask & k_mask]

    d_centroids = jnp.ones_like(lh_centroids) * d_mag
    # d_centroids = d_centroids.at[:, -1].set(d_z)

    return lh_centroids, d_centroids


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
    return dmag - 0.1, Nbin


def enlarge_lh_bins(dataset, lh_centroids, Nmax, dmag=D_MAG, dz=D_Z):
    N_data_lh = []
    dmag_centroids = []

    for lh_centroid in lh_centroids:
        dmag_centroid, Nbin = modulate_dmag(dataset, lh_centroid, Nmax, dmag)
        dmag_centroids.append(dmag_centroid)
        N_data_lh.append(Nbin)

    dmag_centroids = jnp.array(dmag_centroids)
    d_centroids = dmag_centroids.reshape(dmag_centroids.size, 1)

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)

    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )

    return N_data_lh, d_centroids


def get_feniks_data(
    drn,
    ran_key,
    ssp_data,
    lh_sig=LH_SIG,
    lh_n_centroids=LH_N_CENTROIDS,
    z_min=FENIKS_Z_MIN,
    z_max=FENIKS_Z_MAX,
    mag_thresh=FENIKS_MAGK_THRESH,
    sky_area_degsq=FENIKS_AREA_DEG2,
    num_halos=100,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    lc_sky_area_degsq=100,
    n_z_phot_table=120,
    phot=PHOT,
    zout=ZOUT,
    translate=TRANSLATE,
    filter_info=FILTER_INFO,
    tcurves_file=TCURVES_FILE,
    enlarge_dmag=True,
    N_floor=FENIKS_N_FLOOR,
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

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_ri = hsc_r - hsc_i
    hsc_iz = hsc_i - hsc_z
    hsc_video_zY = hsc_z - video_Y
    video_uds_YJ = video_Y - uds_J
    uds_JH = uds_J - uds_H
    uds_HK = uds_H - uds_K

    # stack dataset
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
    zout = zout[z_mask]

    lh_centroids, d_centroids = get_lh_centroids(
        dataset,
        z_min,
        z_max,
        mag_thresh,
    )

    # generate lc
    z_phot_table = 10 ** jnp.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)
    lc_args = (
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

    lc_data = generate_lc_data(*lc_args)

    # run initial diffndhist with fixed dmag
    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )
    vol_mpc3 = zbin_volume(sky_area_degsq, zlow=z_min, zhigh=z_max).value

    if enlarge_dmag is False:
        lg_n, lg_n_avg_err = n_mag.get_n_data_err(N_data_lh, vol_mpc3, N_floor=N_floor)
        lg_n_data_err_lh = jnp.vstack((lg_n, lg_n_avg_err))

        return FENIKS(
            dataset,
            tcurves,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
            lh_centroids,
            d_centroids,
            lg_n_data_err_lh,
            lc_data,
        )
    else:
        # enlarge dmag
        Nmax = N_data_lh.max()
        print("Nmax: " + str(Nmax))

        # run diffndhist with enlarged dmag
        N_data_lh, d_centroids = enlarge_lh_bins(dataset, lh_centroids, Nmax)

        lg_n, lg_n_avg_err = n_mag.get_n_data_err(N_data_lh, vol_mpc3)
        lg_n_data_err_lh = jnp.vstack((lg_n, lg_n_avg_err))

        return FENIKS(
            dataset,
            tcurves,
            mag_columns,
            mag_thresh_column,
            mag_thresh,
            frac_cat,
            lh_centroids,
            d_centroids,
            lg_n_data_err_lh,
            lc_data,
        )
