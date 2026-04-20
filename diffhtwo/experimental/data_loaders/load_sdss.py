from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky.mass_functions import mc_hosts
from DisCoWebS.data_loader import sdss_loader as sdl
from dsps.data_loaders import load_transmission_curve

from .. import diffndhist, n_mag
from ..defaults import (
    DATASET,
    SDSS_AREA_DEG2,
    SDSS_FRAC_CAT,
    SDSS_MAGR_THRESH,
    SDSS_Z_MAX,
    SDSS_Z_MIN,
)
from ..latin_hypercube import latin_hypercube as lh
from ..utils import generate_lc_data, zbin_volume

SDSS = namedtuple("SDSS", DATASET._fields)

D_MAG = 0.3
D_Z = 0.05
LH_N_CENTROIDS = 2500
LH_SIG = 2.5
SDSS_N_FLOOR = 0.5


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

    msk_is_not_outlier = sdl.get_color_outlier_mask(sdss, sdl.SDSS_MAG_NAMES)
    sdss = sdss[msk_is_not_outlier]

    return sdss


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
    mu[1] = mu[1] - 0.1
    mu[4] = mu[4] - 0.2
    mu[5] = mu[5] - 0.02
    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, lh_sig, lh_n_centroids, seed=None
    )

    redshift_mask = (lh_centroids[:, 5] > z_min) & (lh_centroids[:, 5] < z_max)
    r_mask = lh_centroids[:, 4] < mag_thresh
    lh_centroids = lh_centroids[redshift_mask & r_mask]

    d_centroids = jnp.ones_like(lh_centroids) * d_mag
    d_centroids = d_centroids.at[:, -1].set(d_z)

    return lh_centroids, d_centroids


def get_sdss_data(
    drn,
    ran_key,
    ssp_data,
    z_min=SDSS_Z_MIN,
    z_max=SDSS_Z_MAX,
    mag_thresh=SDSS_MAGR_THRESH,
    frac_cat=SDSS_FRAC_CAT,
    sky_area_degsq=SDSS_AREA_DEG2,
    num_halos=100,
    lgmp_min=10.0,
    lgmp_max=mc_hosts.LGMH_MAX,
    lc_sky_area_degsq=100,
    n_z_phot_table=15,
    N_floor=SDSS_N_FLOOR,
):
    sdss = load_sdss_cuts_applied(drn)

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

    sdss_ug = sdss_u - sdss_g
    sdss_gr = sdss_g - sdss_r
    sdss_ri = sdss_r - sdss_i
    sdss_iz = sdss_i - sdss_z

    dataset = np.vstack((sdss_ug, sdss_gr, sdss_ri, sdss_iz, sdss_r, sdss_redshift)).T

    lh_centroids, d_centroids = get_lh_centroids(
        dataset,
        z_min,
        z_max,
        mag_thresh,
    )

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)

    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )

    vol_mpc3 = zbin_volume(sky_area_degsq, zlow=z_min, zhigh=z_max).value

    lg_n, lg_n_avg_err = n_mag.get_n_data_err(N_data_lh, vol_mpc3, N_floor=N_floor)
    lg_n_data_err_lh = jnp.vstack((lg_n, lg_n_avg_err))

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

    return SDSS(
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
