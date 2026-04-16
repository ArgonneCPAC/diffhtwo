from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from DisCoWebS.data_loader import sdss_loader as sdl
from dsps.data_loaders import load_transmission_curve

from .. import diffndhist, n_mag
from ..defaults import SDSS_AREA_DEG2, SDSS_MAGR_THRESH, SDSS_Z_MAX, SDSS_Z_MIN
from ..latin_hypercube import latin_hypercube as lh
from ..utils import zbin_volume

D_MAG = 0.1
D_Z = 0.05
N_CENTROIDS = 2500
SIG = 2.5

SDSS = namedtuple(
    "SDSS", ["dataset", "lh_centroids", "d_centroids", "lg_n_data_err_lh"]
)


def get_sdss_data(drn):
    sdss = sdl.load_sdss_wrapper(drn=drn)

    sdss_filters_to_use = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

    tcurves = []
    for bn_pat in sdss_filters_to_use:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*", drn=drn + "/filters")
        tcurves.append(tcurve)

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

    mu = np.mean(dataset, axis=0)
    mu[4] = mu[4] - 0.4
    mu[5] = mu[5] - 0.01
    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(mu, cov, SIG, N_CENTROIDS, seed=None)

    redshift_mask = (lh_centroids[:, 5] > SDSS_Z_MIN) & (
        lh_centroids[:, 5] < SDSS_Z_MAX
    )
    r_mask = lh_centroids[:, 4] < SDSS_MAGR_THRESH
    lh_centroids = lh_centroids[redshift_mask & r_mask]

    d_centroids = jnp.ones_like(lh_centroids) * D_MAG
    d_centroids = d_centroids.at[:, -1].set(D_Z)

    dataset_sig = jnp.zeros(lh_centroids.shape) + (d_centroids / 2)

    N_data_lh = diffndhist.tw_ndhist(
        dataset,
        dataset_sig,
        lh_centroids - (d_centroids / 2),
        lh_centroids + (d_centroids / 2),
    )

    vol_mpc3 = zbin_volume(SDSS_AREA_DEG2, zlow=SDSS_Z_MIN, zhigh=SDSS_Z_MAX).value

    lg_n, lg_n_avg_err = n_mag.get_n_data_err(N_data_lh, vol_mpc3)
    lg_n_data_err_lh = jnp.vstack((lg_n, lg_n_avg_err))

    return SDSS(dataset, lh_centroids, d_centroids, lg_n_data_err_lh)
