from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from DisCoWebS.data_loader import sdss_loader as sdl
from dsps.data_loaders import load_transmission_curve

from .. import diffndhist
from ..defaults import (
    DATASET,
    SDSS_AREA_DEG2,
    SDSS_MAGR_THRESH,
    SDSS_Z_MAX,
    SDSS_Z_MIN,
)
from ..latin_hypercube import latin_hypercube as lh

SDSS = namedtuple("SDSS", DATASET._fields)

LH_N_CENTROIDS = 20_000
LH_SIG = 3.5
LH_D_MAG = 0.2
LH_D_Z = 0.01


def apply_ra_dec_cut(sdss, ra_min=120, ra_max=240, dec_min=0, dec_max=60):
    return sdss[
        (sdss["ra"] > ra_min)
        & (sdss["ra"] < ra_max)
        & (sdss["dec"] > dec_min)
        & (sdss["dec"] < dec_max)
    ]


def load_sdss_cuts_applied(drn):
    sdss = sdl.load_sdss_cat(drn)

    sdss = apply_ra_dec_cut(sdss)

    # implement r <= 17.6
    mag_thresh_mask = sdss["modelMag_r"] <= SDSS_MAGR_THRESH
    sdss = sdss[mag_thresh_mask]
    N_obj_pre_outlier_cut = len(sdss)

    msk_is_not_outlier = sdl.get_color_outlier_mask(sdss, sdl.SDSS_MAG_NAMES)
    sdss = sdss[msk_is_not_outlier]

    frac_cat = len(sdss) / N_obj_pre_outlier_cut

    return sdss, frac_cat


def refresh_lh_centroids(DATASET):
    lh_centroids, d_centroids = get_lh_centroids(DATASET.dataset)

    # run initial diffndhist with fixed dmag
    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)
    lh_centroids_lo = lh_centroids - (d_centroids / 2)
    lh_centroids_hi = lh_centroids + (d_centroids / 2)
    N_data_lh = diffndhist.tw_ndhist(
        DATASET.dataset,
        dataset_sig,
        lh_centroids_lo,
        lh_centroids_hi,
    )

    DATASET = DATASET._replace(
        lh_centroids=lh_centroids, d_centroids=d_centroids, N_data=N_data_lh
    )

    return DATASET


def get_lh_centroids(dataset):
    mu = np.mean(dataset, axis=0)
    mu[1] = mu[1] - 0.1  #
    mu[-2] = mu[-2] - 1.8  # r
    mu[-1] = mu[-1] - 0.02  # redshift
    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, LH_SIG, LH_N_CENTROIDS, seed=None
    )

    redshift_mask = (lh_centroids[:, -1] > (SDSS_Z_MIN + (LH_D_Z / 2))) & (
        lh_centroids[:, -1] < (SDSS_Z_MAX - (LH_D_Z / 2))
    )
    r_mask = lh_centroids[:, -2] <= SDSS_MAGR_THRESH
    lh_centroids = lh_centroids[redshift_mask & r_mask]

    # redshift = [0.02, 0.065, 0.11, 0.155, 0.2]
    # r_mins = [12, 13.5, 14.5, 15.3, 16]
    # coeffs = np.polyfit(redshift, r_mins, deg=2)
    # r_min = np.poly1d(coeffs)
    # r_complete = lh_centroids[:, -2] > r_min(lh_centroids[:, -1])
    # lh_centroids = lh_centroids[r_complete]

    d_centroids = jnp.ones_like(lh_centroids) * LH_D_MAG
    d_centroids = d_centroids.at[:, -1].set(LH_D_Z)

    return lh_centroids, d_centroids


def get_sdss_data(
    drn,
    ran_key,
    ssp_data,
):
    sdss, frac_cat = load_sdss_cuts_applied(drn)

    sdss_filters = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    tcurves = []
    for bn_pat in sdss_filters:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*", drn=drn + "/filters")
        tcurves.append(tcurve)
    mag_columns = [2]
    mag_thresh_column = 2

    sdss_u = sdss["modelMag_u"].data
    sdss_g = sdss["modelMag_g"].data
    sdss_r = sdss["modelMag_r"].data
    sdss_i = sdss["modelMag_i"].data
    sdss_z = sdss["modelMag_z"].data
    sdss_redshift = sdss["z"].data

    mags = np.vstack((sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, sdss_redshift)).T

    sdss_ug = sdss_u - sdss_g
    sdss_gr = sdss_g - sdss_r
    sdss_ri = sdss_r - sdss_i
    sdss_iz = sdss_i - sdss_z

    dataset = np.vstack((sdss_ug, sdss_gr, sdss_ri, sdss_iz, sdss_r, sdss_redshift)).T

    lh_centroids, d_centroids = get_lh_centroids(dataset)

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

    return SDSS(
        dataset,
        mags,
        tcurves,
        mag_columns,
        mag_thresh_column,
        SDSS_MAGR_THRESH,
        frac_cat,
        lh_centroids,
        d_centroids,
        N_data_lh,
        SDSS_AREA_DEG2,
        LH_D_MAG,
        LH_D_Z,
    )
