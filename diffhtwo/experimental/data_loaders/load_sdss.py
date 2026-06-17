from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem
from DisCoWebS.data_loader import sdss_loader as sdl
from dsps.data_loaders import load_transmission_curve

from ..defaults import (
    SDSS_AREA_DEG2,
    SDSS_MAGR_THRESH,
    SDSS_Z_MAX,
    SDSS_Z_MIN,
    AppMagFunc,
    ColorColor,
    ColorCondMag,
    FilterInfo,
    MagColor,
)
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from .N_utils import get_N_1d, get_N_2d

Sdss = namedtuple(
    "Sdss",
    [
        "dataset",
        "col_idx",
        "mag_idx",
        "dataset_dim_labels",
        "mags",
        "mags_labels",
        "colors",
        "app_mag_funcs",
        "fine_zbins",
        "filter_info",
        "frac_cat",
        "lh_centroids",
        "d_centroids",
        "N_data",
        "lh_dmag",
        "lh_dz",
        "data_sky_area_degsq",
    ],
)


LH_N_CENTROIDS = 20_000
LH_SIG = 3.5
LH_D_Z = 0.05


def apply_ra_dec_cut(sdss, ra_min=120, ra_max=240, dec_min=0, dec_max=60):
    return sdss[
        (sdss["ra"] > ra_min)
        & (sdss["ra"] < ra_max)
        & (sdss["dec"] > dec_min)
        & (sdss["dec"] < dec_max)
    ]


def load_sdss_cuts_applied(drn, sdss_mag_thresh):
    sdss = sdl.load_sdss_cat(drn)

    sdss = apply_ra_dec_cut(sdss)

    # implement r <= 17.6
    mag_thresh_mask = (
        (sdss["modelMag_u"] < sdss_mag_thresh.sdss_u)
        & (sdss["modelMag_g"] < sdss_mag_thresh.sdss_g)
        & (sdss["modelMag_r"] < sdss_mag_thresh.sdss_r)
        & (sdss["modelMag_i"] < sdss_mag_thresh.sdss_i)
        & (sdss["modelMag_z"] < sdss_mag_thresh.sdss_z)
    )
    sdss = sdss[mag_thresh_mask]
    N_obj_pre_outlier_cut = len(sdss)

    msk_is_not_outlier = sdl.get_color_outlier_mask(sdss, sdl.SDSS_MAG_NAMES)
    sdss = sdss[msk_is_not_outlier]

    frac_cat = len(sdss) / N_obj_pre_outlier_cut

    return sdss, frac_cat


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

    mu[-3] = mu[-3] - 0.5  # u
    mu[-2] = mu[-2] - 0.5  # r
    # mu[-1] = mu[-1] 0.02  # redshift

    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, LH_SIG, LH_N_CENTROIDS, seed=None
    )

    redshift_mask = (lh_centroids[:, -1] > (SDSS_Z_MIN + (LH_D_Z / 2))) & (
        lh_centroids[:, -1] < (SDSS_Z_MAX - (LH_D_Z / 2))
    )
    r_mask = lh_centroids[:, -2] <= SDSS_MAGR_THRESH
    lh_centroids = lh_centroids[redshift_mask & r_mask]

    redshift = [0.02, 0.065, 0.11, 0.155, 0.2]
    r_mins = [12, 12.5, 14, 14.5, 15]
    coeffs = np.polyfit(redshift, r_mins, deg=2)
    r_min = np.poly1d(coeffs)
    r_bright = lh_centroids[:, -2] > r_min(lh_centroids[:, -1])
    lh_centroids = lh_centroids[r_bright]

    d_centroids = jnp.ones_like(lh_centroids) * lh_d_mag
    d_centroids = d_centroids.at[:, -1].set(LH_D_Z)

    return lh_centroids, d_centroids


def get_sdss_data(
    drn,
    ran_key,
    ssp_data,
    lh_d_mag=0.1,
    num_halos_coarse_zbins=150,
    num_halos_fine_zbins=250,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
):
    sdss_mag_thresh = SdssFilters(
        sdss_u=19.7,
        sdss_g=18.0,
        sdss_r=SDSS_MAGR_THRESH,
        sdss_i=17.0,
        sdss_z=17.0,
    )
    sdss, frac_cat = load_sdss_cuts_applied(drn, sdss_mag_thresh)

    tcurves = []
    for bn_pat in SdssFilters._fields:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*", drn=drn + "/sdss_filters")
        tcurves.append(tcurve)
    filter_info = FilterInfo(sdss_mag_thresh, tcurves)

    sdss_u = sdss["modelMag_u"].data
    sdss_g = sdss["modelMag_g"].data
    sdss_r = sdss["modelMag_r"].data
    sdss_i = sdss["modelMag_i"].data
    sdss_z = sdss["modelMag_z"].data
    sdss_redshift = sdss["z"].data

    mags = np.vstack((sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, sdss_redshift)).T

    # derive colors from mags
    sdss_ug = sdss_u - sdss_g
    sdss_gr = sdss_g - sdss_r
    sdss_ur = sdss_u - sdss_r
    sdss_ri = sdss_r - sdss_i
    sdss_iz = sdss_i - sdss_z

    # stack colors_mag
    dataset = np.vstack((sdss_ug, sdss_gr, sdss_ri, sdss_iz, sdss_r, sdss_redshift)).T
    dataset_dim_labels = [
        r"$u - g$",
        r"$g - r$",
        r"$r - i$",
        r"$i - z$",
        r"$r$",
        r"$redshift$",
    ]
    mag_labels = [r"$u$", r"$g$", r"$r$", r"$i$", r"$z$"]
    col_idx_lh_dim = [
        [0, 1],  # u - g
        [1, 2],  # g - r
        [2, 3],  # r - i
        [3, 4],  # i - z
    ]
    mag_idx_lh_dim = [2]  # r
    ##############################################################################
    # prepare 2D and 1D color spaces in coarse z-bins for fitting
    zbins = np.array(
        [
            [0.02, 0.1],
            [0.1, 0.2],
        ]
    )
    ##############################################################################
    Colors = namedtuple(
        "Colors",
        ["z_min", "z_max", "lc_data", "ur_ri", "gr_ri", "ur", "ri", "r_ur", "r_ri"],
    )
    # 2D (u - r, r - i)
    Ur_ri = namedtuple("Ur_ri", ColorColor._fields)

    # 2D (g - r, r - i)
    Gr_ri = namedtuple("Gr_ri", ColorColor._fields)

    # 2D (r, u - r)
    R_ur = namedtuple("R_ur", MagColor._fields)

    # 2D (r, r - i)
    R_ri = namedtuple("R_ri", MagColor._fields)

    # 1D (u - r | r)
    Ur_condr = namedtuple("Ug_condK", ColorCondMag._fields)

    # 1D (r - i | r)
    Ri_condr = namedtuple("Ri_condr", ColorCondMag._fields)

    colors = []
    for zbin in range(0, len(zbins)):
        z_min = zbins[zbin][0]
        z_max = zbins[zbin][1]

        z_phot_table = 10 ** jnp.linspace(
            jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
        )
        lc_args = (
            ran_key,
            num_halos_coarse_zbins,
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

        z_sel = (sdss_redshift > z_min) & (sdss_redshift <= z_max)

        # 2D (u - r, r - i)
        N_ur_ri, sig_ur_ri, bin_lo_ur_ri, bin_hi_ur_ri = get_N_2d(
            sdss_ur[z_sel], sdss_ri[z_sel]
        )
        col_idx = [0, 2, 2, 3]
        ur_ri = Ur_ri(col_idx, sig_ur_ri, bin_lo_ur_ri, bin_hi_ur_ri, N_ur_ri, True)

        # 2D (g - r, r - i)
        N_gr_ri, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri = get_N_2d(
            sdss_gr[z_sel], sdss_ri[z_sel]
        )
        col_idx = [1, 2, 2, 3]
        gr_ri = Gr_ri(col_idx, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri, N_gr_ri, True)

        # 1D (u - r | r)
        rbins = np.arange(sdss_r[z_sel].min(), sdss_r[z_sel].max(), 2)

        col_idx = [0, 2]
        cond_idx = 2
        ur = []
        for r in range(len(rbins) - 1):
            r_sel = (sdss_r[z_sel] > rbins[r]) & (sdss_r[z_sel] <= rbins[r + 1])
            N_1d_ur, sig_ur, bin_lo_ur, bin_hi_ur = get_N_1d(sdss_ur[z_sel][r_sel])
            ur.append(
                Ur_condr(
                    col_idx,
                    cond_idx,
                    rbins[r],
                    rbins[r + 1],
                    sig_ur,
                    bin_lo_ur,
                    bin_hi_ur,
                    N_1d_ur,
                    True,
                )
            )

        # 1D (r - i | r)
        col_idx = [2, 3]
        cond_idx = 2
        ri = []
        for r in range(len(rbins) - 1):
            r_sel = (sdss_r[z_sel] > rbins[r]) & (sdss_r[z_sel] <= rbins[r + 1])
            N_1d_ri, sig_ri, bin_lo_ri, bin_hi_ri = get_N_1d(sdss_ri[z_sel][r_sel])
            ri.append(
                Ri_condr(
                    col_idx,
                    cond_idx,
                    rbins[r],
                    rbins[r + 1],
                    sig_ri,
                    bin_lo_ri,
                    bin_hi_ri,
                    N_1d_ri,
                    True,
                )
            )

        # 2D (r, u - r)
        N_r_ur, sig_r_ur, bin_lo_r_ur, bin_hi_r_ur = get_N_2d(
            sdss_r[z_sel], sdss_ur[z_sel]
        )
        mag_idx = 2
        col_idx = [0, 2]
        r_ur = R_ur(mag_idx, col_idx, sig_r_ur, bin_lo_r_ur, bin_hi_r_ur, N_r_ur, False)

        # 2D (r, r - i)
        N_r_ri, sig_r_ri, bin_lo_r_ri, bin_hi_r_ri = get_N_2d(
            sdss_r[z_sel], sdss_ri[z_sel]
        )
        mag_idx = 2
        col_idx = [2, 3]
        r_ri = R_ri(mag_idx, col_idx, sig_r_ri, bin_lo_r_ri, bin_hi_r_ri, N_r_ri, False)

        colors.append(Colors(z_min, z_max, lc_data, ur_ri, gr_ri, ur, ri, r_ur, r_ri))

    ##############################################################################
    ##############################################################################
    # prepare 1D app mag funcs in finer z-bins for fitting
    fine_zbins = np.array([[0.02, 0.06], [0.06, 0.1], [0.1, 0.14], [0.14, 0.2]])
    ##############################################################################
    AppMagFuncs = namedtuple(
        "AppMagFuncs",
        ["z_min", "z_max", "lc_data", "u", "g", "r", "i", "z"],
    )
    U = namedtuple("U", AppMagFunc._fields)
    G = namedtuple("G", AppMagFunc._fields)
    R = namedtuple("R", AppMagFunc._fields)
    I = namedtuple("I", AppMagFunc._fields)  # noqa: E741
    Z = namedtuple("Z", AppMagFunc._fields)

    app_mag_funcs = []
    for zbin in range(0, len(fine_zbins)):
        z_min = fine_zbins[zbin][0]
        z_max = fine_zbins[zbin][1]

        z_phot_table = 10 ** jnp.linspace(
            jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
        )
        lc_args = (
            ran_key,
            num_halos_fine_zbins,
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

        z_sel = (sdss_redshift > z_min) & (sdss_redshift <= z_max)

        # 1D (u)
        mag_idx_u = 0
        N_1d_u, sig_u, bin_lo_u, bin_hi_u = get_N_1d(sdss_u[z_sel])
        u = U(mag_idx_u, sig_u, bin_lo_u, bin_hi_u, N_1d_u, True)

        # 1D (g)
        mag_idx_g = 1
        N_1d_g, sig_g, bin_lo_g, bin_hi_g = get_N_1d(sdss_g[z_sel])
        g = G(mag_idx_g, sig_g, bin_lo_g, bin_hi_g, N_1d_g, True)

        # 1D (r)
        mag_idx_r = 2
        N_1d_r, sig_r, bin_lo_r, bin_hi_r = get_N_1d(sdss_r[z_sel])
        r = R(mag_idx_r, sig_r, bin_lo_r, bin_hi_r, N_1d_r, True)

        # 1D (i)
        mag_idx_i = 3
        N_1d_i, sig_i, bin_lo_i, bin_hi_i = get_N_1d(sdss_i[z_sel])
        i = I(mag_idx_i, sig_i, bin_lo_i, bin_hi_i, N_1d_i, True)

        # 1D (z)
        mag_idx_z = 4
        N_1d_z, sig_z, bin_lo_z, bin_hi_z = get_N_1d(sdss_z[z_sel])
        z = Z(mag_idx_z, sig_z, bin_lo_z, bin_hi_z, N_1d_z, True)

        app_mag_funcs.append(AppMagFuncs(z_min, z_max, lc_data, u, g, r, i, z))

    ##############################################################################

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

    return Sdss(
        dataset,
        col_idx_lh_dim,
        mag_idx_lh_dim,
        dataset_dim_labels,
        mags,
        mag_labels,
        colors,
        app_mag_funcs,
        fine_zbins,
        filter_info,
        frac_cat,
        lh_centroids,
        d_centroids,
        N_data_lh,
        lh_d_mag,
        LH_D_Z,
        SDSS_AREA_DEG2,
    )


SdssFilters = namedtuple(
    "SdssFilters",
    [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
    ],
)
