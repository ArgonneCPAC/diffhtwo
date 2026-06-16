import warnings
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from diffsky import diffndhist_lomem
from dsps.data_loaders.defaults import TransmissionCurve
from scipy import optimize

from ..defaults import (
    FENIKS_AREA_DEG2,
    FENIKS_MAGK_THRESH,
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
    Dataset,
    DatasetLH,
    FeniksFilters,
    FilterInfo,
)
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from ..utils import add_random_rows, load_feniks_tcurve
from . import N_utils

BASE_PATH = Path(__file__).resolve().parent.parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"

Feniks = namedtuple("Feniks", Dataset._fields)
FeniksLH = namedtuple("FeniksLH", DatasetLH._fields)

LH_SIG = 3.0
LH_N_CENTROIDS = 75_000

LH_D_Z = 0.3


def _power_law(x, A, B):
    return A * (x**B)


def _get_mag_thresh(mag, completeness=0.9, power_law_limit=24):
    mag_bin_edges = np.arange(22, 28, 0.2)
    mag_bin_centers = (mag_bin_edges[1:] + mag_bin_edges[:-1]) / 2

    N, _ = np.histogram(mag, bins=mag_bin_edges)
    lg_N = np.log10(N)

    mag_sel = mag_bin_centers < power_law_limit
    copt, ccov = optimize.curve_fit(_power_law, mag_bin_centers[mag_sel], lg_N[mag_sel])

    lg_N_modeled = _power_law(mag_bin_centers, copt[0], copt[1])
    ratio = lg_N / lg_N_modeled

    mag_sel_faint = mag_bin_centers >= power_law_limit
    mag_bin_centers = mag_bin_centers[mag_sel_faint]
    ratio = ratio[mag_sel_faint]

    for m in range(0, len(mag_bin_centers)):
        if ratio[m] < completeness:
            mag_thresh = mag_bin_centers[m - 1]
            break
    return np.round(mag_thresh, 1)


def get_mag_ab(phot_table, col_name, ZP=25):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_ab = -2.5 * np.log10(phot_table[col_name]) + ZP

    mag_ab[~np.isfinite(mag_ab)] = -99.0
    mag_ab = mag_ab.data

    # mag_thresh = _get_mag_thresh(mag_ab[mag_ab != -99])

    return mag_ab


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

    mu[0] = mu[0] + 0.4  # u - g
    # mu[1] = mu[1] + 0.0  # g - r
    # mu[2] = mu[2] + 0.0  # r - i
    # mu[3] = mu[3] + 0.1  # z - Y
    # mu[4] = mu[4] + 0.15  # z - Y
    # mu[5] = mu[5] + 0.0  # Y - J
    # mu[6] = mu[6] + 0.0  # J - H
    mu[-3] = mu[-3] - 1.0  # u

    mu[-2] = mu[-2] - 1.0  # K
    # mu[-1] = mu[-1] + 0.5  # redshift

    cov = np.cov(dataset.T)

    lh_centroids = lh.latin_hypercube_from_cov(
        mu, cov, LH_SIG, LH_N_CENTROIDS, seed=None
    )

    redshift_mask = (lh_centroids[:, -1] > (FENIKS_Z_MIN + (LH_D_Z / 2))) & (
        lh_centroids[:, -1] < (FENIKS_Z_MAX - (LH_D_Z / 2))
    )
    k_mask = lh_centroids[:, -2] < FENIKS_MAGK_THRESH
    u_mask = lh_centroids[:, -3] < 24.9
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


FeniksFiltersLH = namedtuple(
    "FeniksFiltersLH",
    [
        "MegaCam_uS",
        "HSC_G",
        "HSC_R",
        "HSC_I",
        "HSC_Z",
        "UDS_J",
        "UDS_H",
        "UDS_K",
    ],
)


def get_feniks_data_lh(
    drn,
    ran_key,
    ssp_data,
    lh_d_mag=0.6,
    phot=PHOT,
    zout=ZOUT,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    add_random_rows_for_testing=False,
):
    # Transmission curves and filter mag thresholds
    tcurves = []
    for feniks_filter in FeniksFiltersLH._fields:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    drn_path = Path(drn)
    phot = ascii.read(drn_path / phot)
    zout = ascii.read(drn_path / zout)

    if add_random_rows_for_testing:
        phot = add_random_rows(phot, N=10000)
        zout = add_random_rows(zout, N=10000)

    # get mags
    megacam_uS = get_mag_ab(phot, "fcol_MegaCam_uS")
    hsc_g = get_mag_ab(phot, "fcol_HSC_G")
    hsc_r = get_mag_ab(phot, "fcol_HSC_R")
    hsc_i = get_mag_ab(phot, "fcol_HSC_I")
    hsc_z = get_mag_ab(phot, "fcol_HSC_Z")
    uds_J = get_mag_ab(phot, "fcol_UDS_J")
    uds_H = get_mag_ab(phot, "fcol_UDS_H")
    uds_K = get_mag_ab(phot, "fcol_UDS_K")

    feniks_mag_thresh = FeniksFiltersLH(
        MegaCam_uS=24.9,
        HSC_G=25.1,
        HSC_R=25.3,
        HSC_I=25.1,
        HSC_Z=24.9,
        UDS_J=24.5,
        UDS_H=24.3,
        UDS_K=FENIKS_MAGK_THRESH,
    )

    filter_info = FilterInfo(feniks_mag_thresh, tcurves)

    # get mag thresh cuts
    mag_thresh = (
        (megacam_uS < feniks_mag_thresh.MegaCam_uS)
        & (hsc_g < feniks_mag_thresh.HSC_G)
        & (hsc_r < feniks_mag_thresh.HSC_R)
        & (hsc_i < feniks_mag_thresh.HSC_I)
        & (hsc_z < feniks_mag_thresh.HSC_Z)
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
    uds_J = uds_J[mag_thresh]
    uds_H = uds_H[mag_thresh]
    uds_K = uds_K[mag_thresh]

    n_gals_pre_cuts = len(zout)

    # remove mags with bad data in any of the bands
    clean = (
        (megacam_uS != -99)
        & (hsc_g != -99)
        & (hsc_r != -99)
        & (hsc_i != -99)
        & (hsc_z != -99)
        & (uds_J != -99)
        & (uds_H != -99)
        & (uds_K != -99)
        & (zout["z_phot"] >= 0)
    )

    phot = phot[clean]
    zout = zout[clean]
    megacam_uS = megacam_uS[clean]
    hsc_g = hsc_g[clean]
    hsc_r = hsc_r[clean]
    hsc_i = hsc_i[clean]
    hsc_z = hsc_z[clean]
    uds_J = uds_J[clean]
    uds_H = uds_H[clean]
    uds_K = uds_K[clean]

    n_gals_post_cuts = len(zout)
    frac_cat = n_gals_post_cuts / n_gals_pre_cuts

    mags = np.vstack(
        (
            megacam_uS,
            hsc_g,
            hsc_r,
            hsc_i,
            hsc_z,
            uds_J,
            uds_H,
            uds_K,
            zout["z_phot"],
        )
    ).T

    mags_labels = [
        r"$uS_{MegaCam}$",
        r"$g_{HSC}$",
        r"$r_{HSC}$",
        r"$i_{HSC}$",
        r"$z_{HSC}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
        r"$redshift$",
    ]

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_ri = hsc_r - hsc_i
    hsc_iz = hsc_i - hsc_z
    hsc_uds_zJ = hsc_z - uds_J
    uds_JH = uds_J - uds_H
    hsc_uds_rK = hsc_r - uds_K

    # stack colors_mag
    dataset = np.vstack(
        (
            megacam_hsc_uSg,
            hsc_gr,
            hsc_ri,
            hsc_iz,
            hsc_uds_zJ,
            uds_JH,
            hsc_uds_rK,
            megacam_uS,
            uds_K,
            zout["z_phot"],
        )
    ).T

    col_idx = [
        [0, 1],  # u - g
        [1, 2],  # g - r
        [2, 3],  # r - i
        [3, 4],  # i - z
        [4, 5],  # z - J
        [5, 6],  # J - H
        [2, 7],  # r - K
    ]
    mag_idx = [
        0,  # u
        7,  # K
    ]
    dataset_dim_labels = [
        r"$uS_{MegaCam} - g_{HSC}$",
        r"$g_{HSC} - r_{HSC}$",
        r"$r_{HSC} - i_{HSC}$",
        r"$i_{HSC} - z_{HSC}$",
        r"$z_{HSC} - J_{UDS}$",
        r"$J_{UDS} - H_{UDS}$",
        r"$r_{HSC} - K_{UDS}$",
        r"$uS_{MegaCam}$",
        r"$K_{UDS}$",
        r"$redshift$",
    ]

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

    return FeniksLH(
        dataset,
        col_idx,
        mag_idx,
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


def get_feniks_data(
    drn,
    ran_key,
    ssp_data,
    lh_d_mag=0.6,
    phot=PHOT,
    zout=ZOUT,
    num_halos_coarse_zbins=250,
    num_halos_fine_zbins=150,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    add_random_rows_for_testing=False,
):
    # Transmission curves and filter mag thresholds
    tcurves = []
    for feniks_filter in FeniksFilters._fields:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    drn_path = Path(drn)
    phot = ascii.read(drn_path / phot)
    zout = ascii.read(drn_path / zout)

    if add_random_rows_for_testing:
        phot = add_random_rows(phot, N=10000)
        zout = add_random_rows(zout, N=10000)

    # get mags
    megacam_uS = get_mag_ab(phot, "fcol_MegaCam_uS")
    hsc_g = get_mag_ab(phot, "fcol_HSC_G")
    hsc_r = get_mag_ab(phot, "fcol_HSC_R")
    hsc_i = get_mag_ab(phot, "fcol_HSC_I")
    # nb816 = get_mag_ab(phot, "fcol_NB0816")
    hsc_z = get_mag_ab(phot, "fcol_HSC_Z")
    # nb921 = get_mag_ab(phot, "fcol_NB0921")
    uds_J = get_mag_ab(phot, "fcol_UDS_J")
    uds_H = get_mag_ab(phot, "fcol_UDS_H")
    uds_K = get_mag_ab(phot, "fcol_UDS_K")

    feniks_mag_thresh = FeniksFilters(
        MegaCam_uS=24.9,
        HSC_G=25.1,
        HSC_R=25.3,
        HSC_I=25.1,
        # NB0816=25.3,
        HSC_Z=24.9,
        # NB0921=25.3,
        UDS_J=24.5,
        UDS_H=24.3,
        UDS_K=FENIKS_MAGK_THRESH,
    )

    filter_info = FilterInfo(feniks_mag_thresh, tcurves)

    # get mag thresh cuts
    mag_thresh = (
        (megacam_uS < feniks_mag_thresh.MegaCam_uS)
        & (hsc_g < feniks_mag_thresh.HSC_G)
        & (hsc_r < feniks_mag_thresh.HSC_R)
        & (hsc_i < feniks_mag_thresh.HSC_I)
        # & (nb816 < feniks_mag_thresh.NB0816)
        & (hsc_z < feniks_mag_thresh.HSC_Z)
        # & (nb921 < feniks_mag_thresh.NB0921)
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
    # nb816 = nb816[mag_thresh]
    hsc_z = hsc_z[mag_thresh]
    # nb921 = nb921[mag_thresh]
    uds_J = uds_J[mag_thresh]
    uds_H = uds_H[mag_thresh]
    uds_K = uds_K[mag_thresh]

    n_gals_pre_cuts = len(zout)

    # remove mags with bad data in any of the bands
    clean = (
        (megacam_uS != -99)
        & (hsc_g != -99)
        & (hsc_r != -99)
        & (hsc_i != -99)
        # & (nb816 != -99)
        & (hsc_z != -99)
        # & (nb921 != -99)
        & (uds_J != -99)
        & (uds_H != -99)
        & (uds_K != -99)
        & (zout["z_phot"] >= 0)
    )

    phot = phot[clean]
    zout = zout[clean]
    megacam_uS = megacam_uS[clean]
    hsc_g = hsc_g[clean]
    hsc_r = hsc_r[clean]
    hsc_i = hsc_i[clean]
    # nb816 = nb816[clean]
    hsc_z = hsc_z[clean]
    # nb921 = nb921[clean]
    uds_J = uds_J[clean]
    uds_H = uds_H[clean]
    uds_K = uds_K[clean]

    n_gals_post_cuts = len(zout)
    frac_cat = n_gals_post_cuts / n_gals_pre_cuts

    mags = np.vstack(
        (
            megacam_uS,
            hsc_g,
            hsc_r,
            hsc_i,
            # nb816,
            hsc_z,
            # nb921,
            uds_J,
            uds_H,
            uds_K,
            zout["z_phot"],
        )
    ).T

    mags_labels = [
        r"$uS_{MegaCam}$",
        r"$g_{HSC}$",
        r"$r_{HSC}$",
        r"$i_{HSC}$",
        # r"$NB816_{HSC}$",
        r"$z_{HSC}$",
        # r"$NB921_{HSC}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
        r"$redshift$",
    ]

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_rz = hsc_r - hsc_z
    hsc_ri = hsc_r - hsc_i
    # hsc_i816 = hsc_i - nb816
    hsc_iz = hsc_i - hsc_z
    # hsc_z921 = hsc_z - nb921
    hsc_uds_zJ = hsc_z - uds_J
    uds_JH = uds_J - uds_H
    hsc_uds_rK = hsc_r - uds_K

    # stack colors_mag
    dataset = np.vstack(
        (
            megacam_hsc_uSg,
            hsc_gr,
            hsc_ri,
            # hsc_i816,
            hsc_iz,
            # hsc_z921,
            hsc_uds_zJ,
            uds_JH,
            hsc_uds_rK,
            megacam_uS,
            uds_K,
            zout["z_phot"],
        )
    ).T

    dataset_dim_labels = [
        r"$uS_{MegaCam} - g_{HSC}$",
        r"$g_{HSC} - r_{HSC}$",
        r"$r_{HSC} - i_{HSC}$",
        # r"$i_{HSC} - NB816_{HSC}$",
        r"$i_{HSC} - z_{HSC}$",
        # r"$z_{HSC} - NB921_{HSC}$",
        r"$z_{HSC} - J_{UDS}$",
        r"$J_{UDS} - H_{UDS}$",
        r"$r_{HSC} - K_{UDS}$",
        r"$uS_{MegaCam}$",
        r"$K_{UDS}$",
        r"$redshift$",
    ]

    ##############################################################################
    # prepare 2D and 1D color spaces in coarse z-bins for fitting
    zbins = np.array(
        [
            [0.2, 0.7],
            [0.7, 1.5],
            [1.5, 2.5],
        ]
    )
    ##############################################################################
    # Z1 spaces:
    # 2D (g - r, r - i)
    # 2D (K, g - r)
    # 2D (K, r - i)
    # 2D (K, J - H)

    colors = []
    Z1 = namedtuple(
        "Z1",
        [
            "z_min",
            "z_max",
            "lc_data",
            "gr_ri",
            "ug",
            "ri",
            "iz",
            "jh",
            "K_ri",
            "K_gr",
            "K_JH",
        ],
    )
    zbin = 0
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

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)

    # 2D (g - r, r - i)
    gr_ri = N_utils.get_colorcolor_space(
        "Gr_ri", hsc_gr, hsc_ri, ["HSC_G", "HSC_R", "HSC_R", "HSC_I"], z_sel, fit=True
    )

    # 1D (u - g | K)
    ug = N_utils.get_color_cond_space_list(
        "Ug_condK",
        megacam_hsc_uSg,
        uds_K,
        ["MegaCam_uS", "HSC_G"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 1D (r − i | K)
    ri = N_utils.get_color_cond_space_list(
        "Ri_condK",
        hsc_ri,
        uds_K,
        ["HSC_R", "HSC_I"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 1D (i − z | K)
    iz = N_utils.get_color_cond_space_list(
        "Iz_condK",
        hsc_iz,
        uds_K,
        ["HSC_I", "HSC_Z"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 1D (J − H | K)
    jh = N_utils.get_color_cond_space_list(
        "JH_condK",
        uds_JH,
        uds_K,
        ["UDS_J", "UDS_H"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 2D (K, r - i)
    K_ri = N_utils.get_mag_color_space(
        "K_ri", uds_K, hsc_ri, "UDS_K", ["HSC_R", "HSC_I"], z_sel, fit=False
    )

    # 2D (K, g - r)
    K_gr = N_utils.get_mag_color_space(
        "K_gr", uds_K, hsc_gr, "UDS_K", ["HSC_G", "HSC_R"], z_sel, fit=False
    )

    # 2D (K, J - H)
    K_JH = N_utils.get_mag_color_space(
        "K_JH", uds_K, uds_JH, "UDS_K", ["UDS_J", "UDS_H"], z_sel, fit=False
    )

    z1 = Z1(z_min, z_max, lc_data, gr_ri, ug, ri, iz, jh, K_ri, K_gr, K_JH)
    colors.append(z1)

    ##############################################################################
    # Z2a spaces:
    # 2D (r - z, z - J)
    # 2D (K, u - g)
    # 2D (K, r - z)
    # 2D (i - NB816, g - r)   -- metallicity vs age
    # 2D (i - NB816, r - K)   -- metallicity vs mass-to-light
    # 1D (i - NB816 | K)      -- metallicity at fixed mass
    # 1D (z - NB921 | K)      -- cross-check metallicity at fixed mass

    # Z2a = namedtuple(
    #     "Z2a",
    #     [
    #         "z_min",
    #         "z_max",
    #         "lc_data",
    #         "rz_zJ",
    #         "ug",
    #         "rz",
    #         "jh",
    #         "K_ug",
    #         "K_rz",
    #         "iNB816_gr",
    #         "iNB816_rK",
    #         "iNB816_condK",
    #         "zNB921_condK",
    #     ],
    # )
    # zbin = 1
    # z_min = zbins[zbin][0]
    # z_max = zbins[zbin][1]

    # z_phot_table = 10 ** jnp.linspace(
    #     jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    # )
    # lc_args = (
    #     ran_key,
    #     num_halos_coarse_zbins,
    #     z_min,
    #     z_max,
    #     lgmp_min,
    #     lgmp_max,
    #     lc_sky_area_degsq,
    #     ssp_data,
    #     tcurves,
    #     z_phot_table,
    # )

    # lc_data = generate_lc_data(*lc_args)

    # z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)

    # # 2D (r - z, z - J)
    # rz_zJ = N_utils.get_colorcolor_space(
    #     "Rz_zJ",
    #     hsc_rz,
    #     hsc_uds_zJ,
    #     ["HSC_R", "HSC_Z", "HSC_Z", "UDS_J"],
    #     z_sel,
    #     fit=True,
    # )

    # # 1D (u - g | K)
    # ug = N_utils.get_color_cond_space_list(
    #     "Ug_condK",
    #     megacam_hsc_uSg,
    #     uds_K,
    #     ["MegaCam_uS", "HSC_G"],
    #     "UDS_K",
    #     z_sel,
    #     cond_dmag=2,
    #     fit=True,
    # )

    # # 1D (r - z | K)
    # rz = N_utils.get_color_cond_space_list(
    #     "Rz_condK",
    #     hsc_rz,
    #     uds_K,
    #     ["HSC_R", "HSC_Z"],
    #     "UDS_K",
    #     z_sel,
    #     cond_dmag=2,
    #     fit=True,
    # )

    # # 1D (J − H | K)
    # jh = N_utils.get_color_cond_space_list(
    #     "JH_condK",
    #     uds_JH,
    #     uds_K,
    #     ["UDS_J", "UDS_H"],
    #     "UDS_K",
    #     z_sel,
    #     cond_dmag=2,
    #     fit=True,
    # )

    # # 2D (K, u - g)
    # K_ug = N_utils.get_mag_color_space(
    #     "K_ug",
    #     uds_K,
    #     megacam_hsc_uSg,
    #     "UDS_K",
    #     ["MegaCam_uS", "HSC_G"],
    #     z_sel,
    #     fit=False,
    # )

    # # 2D (K, r - z)
    # K_rz = N_utils.get_mag_color_space(
    #     "K_rz", uds_K, hsc_rz, "UDS_K", ["HSC_R", "HSC_Z"], z_sel, fit=False
    # )

    # # 2D (i - NB816, g - r)
    # i816_gr = N_utils.get_colorcolor_space(
    #     "I816_gr",
    #     hsc_i816,
    #     hsc_gr,
    #     ["HSC_I", "NB0816", "HSC_G", "HSC_R"],
    #     z_sel,
    #     fit=False,
    # )

    # # 2D (i - NB816, r - K)
    # i816_rK = N_utils.get_colorcolor_space(
    #     "I816_rK",
    #     hsc_i816,
    #     hsc_uds_rK,
    #     ["HSC_I", "NB0816", "HSC_R", "UDS_K"],
    #     z_sel,
    #     fit=False,
    # )

    # # 1D (i - NB816 | K)
    # i816_condK = N_utils.get_color_cond_space_list(
    #     "I816_condK",
    #     hsc_i816,
    #     uds_K,
    #     ["HSC_I", "NB0816"],
    #     "UDS_K",
    #     z_sel,
    #     cond_dmag=2,
    #     fit=False,
    # )

    # # 1D (z - NB921 | K)
    # z921_condK = N_utils.get_color_cond_space_list(
    #     "Z921_condK",
    #     hsc_z921,
    #     uds_K,
    #     ["HSC_Z", "NB0921"],
    #     "UDS_K",
    #     z_sel,
    #     cond_dmag=2,
    #     fit=False,
    # )

    # z2a = Z2a(
    #     z_min,
    #     z_max,
    #     lc_data,
    #     rz_zJ,
    #     ug,
    #     rz,
    #     jh,
    #     K_ug,
    #     K_rz,
    #     i816_gr,
    #     i816_rK,
    #     i816_condK,
    #     z921_condK,
    # )
    # colors.append(z2a)

    ##############################################################################
    # Z2b spaces:
    # 2D (r - z, z - J)
    # 2D (K, u - g)
    # 2D (K, r - z)

    Z2b = namedtuple(
        "Z2b",
        ["z_min", "z_max", "lc_data", "rz_zJ", "ug", "rz", "jh", "K_ug", "K_rz"],
    )
    zbin = 1
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

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)

    # 2D (r - z, z - J)
    rz_zJ = N_utils.get_colorcolor_space(
        "Rz_zJ",
        hsc_rz,
        hsc_uds_zJ,
        ["HSC_R", "HSC_Z", "HSC_Z", "UDS_J"],
        z_sel,
        fit=True,
    )

    # 1D (u - g | K)
    ug = N_utils.get_color_cond_space_list(
        "Ug_condK",
        megacam_hsc_uSg,
        uds_K,
        ["MegaCam_uS", "HSC_G"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 1D (r - z | K)
    rz = N_utils.get_color_cond_space_list(
        "Rz_condK",
        hsc_rz,
        uds_K,
        ["HSC_R", "HSC_Z"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 1D (J − H | K)
    jh = N_utils.get_color_cond_space_list(
        "JH_condK",
        uds_JH,
        uds_K,
        ["UDS_J", "UDS_H"],
        "UDS_K",
        z_sel,
        cond_dmag=2,
        fit=True,
    )

    # 2D (K, u - g)
    K_ug = N_utils.get_mag_color_space(
        "K_ug",
        uds_K,
        megacam_hsc_uSg,
        "UDS_K",
        ["MegaCam_uS", "HSC_G"],
        z_sel,
        fit=False,
    )

    # 2D (K, r - z)
    K_rz = N_utils.get_mag_color_space(
        "K_rz", uds_K, hsc_rz, "UDS_K", ["HSC_R", "HSC_Z"], z_sel, fit=False
    )

    z2b = Z2b(z_min, z_max, lc_data, rz_zJ, ug, rz, jh, K_ug, K_rz)
    colors.append(z2b)

    ##############################################################################
    # Z3 spaces:
    # 2D (z - J, J - H)
    # 2D (u - g, g - r)
    # 2D (K, u - g)
    # 2D (K, g - r)
    # 2D (K, J − H): residual quenching scatter at fixed stellar mass

    Z3 = namedtuple(
        "Z3",
        [
            "z_min",
            "z_max",
            "lc_data",
            "zJ_JH",
            "ug_gr",
            "ug",
            "gr",
            "jh",
            "K_ug",
            "K_gr",
            "K_JH",
        ],
    )
    zbin = 2
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

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)

    # 2D (z - J, J - H)
    zJ_JH = N_utils.get_colorcolor_space(
        "ZJ_JH",
        hsc_uds_zJ,
        uds_JH,
        ["HSC_Z", "UDS_J", "UDS_J", "UDS_H"],
        z_sel,
        fit=True,
    )

    # 2D (u - g, g - r)
    ug_gr = N_utils.get_colorcolor_space(
        "Ug_gr",
        megacam_hsc_uSg,
        hsc_gr,
        ["MegaCam_uS", "HSC_G", "HSC_G", "HSC_R"],
        z_sel,
        fit=True,
    )

    # 1D (u - g | K)
    ug = N_utils.get_color_cond_space_list(
        "Ug_condK",
        megacam_hsc_uSg,
        uds_K,
        ["MegaCam_uS", "HSC_G"],
        "UDS_K",
        z_sel,
        cond_dmag=4,
        fit=True,
    )

    # 1D (g - r | K)
    gr = N_utils.get_color_cond_space_list(
        "Gr_condK",
        hsc_gr,
        uds_K,
        ["HSC_G", "HSC_R"],
        "UDS_K",
        z_sel,
        cond_dmag=4,
        fit=True,
    )

    # 1D (J − H | K)
    jh = N_utils.get_color_cond_space_list(
        "JH_condK",
        uds_JH,
        uds_K,
        ["UDS_J", "UDS_H"],
        "UDS_K",
        z_sel,
        cond_dmag=4,
        fit=True,
    )

    # 2D (K, u - g)
    K_ug = N_utils.get_mag_color_space(
        "K_ug",
        uds_K,
        megacam_hsc_uSg,
        "UDS_K",
        ["MegaCam_uS", "HSC_G"],
        z_sel,
        fit=False,
    )

    # 2D (K, g - r)
    K_gr = N_utils.get_mag_color_space(
        "K_gr", uds_K, hsc_gr, "UDS_K", ["HSC_G", "HSC_R"], z_sel, fit=False
    )

    # 2D (K, J - H)
    K_JH = N_utils.get_mag_color_space(
        "K_JH", uds_K, uds_JH, "UDS_K", ["UDS_J", "UDS_H"], z_sel, fit=False
    )

    z3 = Z3(z_min, z_max, lc_data, zJ_JH, ug_gr, ug, gr, jh, K_ug, K_gr, K_JH)
    colors.append(z3)

    ##############################################################################
    # prepare 1D app mag funcs in finer z-bins for fitting
    fine_zbins = np.array(
        [
            [0.2, 0.5],
            [0.5, 0.7],
            [0.7, 1.0],
            [1.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
        ]
    )
    ##############################################################################
    AppMagFuncs = namedtuple(
        "AppMagFuncs",
        ["z_min", "z_max", "lc_data", "u", "g", "r", "i", "z", "J", "H", "K"],
    )

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

        z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)

        # 1D (u)
        u = N_utils.get_mag_space("U", megacam_uS, "MegaCam_uS", z_sel, fit=True)

        # 1D (g)
        g = N_utils.get_mag_space("G", hsc_g, "HSC_G", z_sel, fit=False)

        # 1D (r)
        r = N_utils.get_mag_space("R", hsc_r, "HSC_R", z_sel, fit=True)

        # 1D (i)
        i = N_utils.get_mag_space("I", hsc_i, "HSC_I", z_sel, fit=False)

        # 1D (z)
        z = N_utils.get_mag_space("Z", hsc_z, "HSC_Z", z_sel, fit=False)

        # 1D (J)
        j = N_utils.get_mag_space("J", uds_J, "UDS_J", z_sel, fit=False)

        # 1D (H)
        h = N_utils.get_mag_space("H", uds_H, "UDS_H", z_sel, fit=False)

        # 1D (K)
        k = N_utils.get_mag_space("K", uds_K, "UDS_K", z_sel, fit=True)

        app_mag_funcs.append(AppMagFuncs(z_min, z_max, lc_data, u, g, r, i, z, j, h, k))

    ##############################################################################

    return Feniks(
        dataset,
        dataset_dim_labels,
        mags,
        mags_labels,
        colors,
        app_mag_funcs,
        fine_zbins,
        filter_info,
        frac_cat,
        FENIKS_AREA_DEG2,
    )


def get_feniks_fitting_data(
    feniks_drn,
    ran_key,
    ssp_data,
    phot=PHOT,
    zout=ZOUT,
    num_halos_coarse_zbins=250,
    num_halos_fine_zbins=150,
    add_random_rows_for_testing=False,
):
    feniks = get_feniks_data(
        feniks_drn,
        ran_key,
        ssp_data,
        phot=phot,
        zout=zout,
        num_halos_coarse_zbins=num_halos_coarse_zbins,
        num_halos_fine_zbins=num_halos_fine_zbins,
        add_random_rows_for_testing=add_random_rows_for_testing,
    )
    remove = {"dataset_dim_labels", "mags_labels"}
    FeniksFitting = namedtuple("Feniks", [f for f in feniks._fields if f not in remove])
    feniks_fitting_data = FeniksFitting(
        **{f: getattr(feniks, f) for f in FeniksFitting._fields}
    )

    return feniks_fitting_data


# FeniksFilters = namedtuple(
#     "FeniksFilters",
#     [
#         "MegaCam_uS",
#         "HSC_G",
#         "HSC_R",
#         "HSC_I",
#         "NB0816",
#         "HSC_Z",
#         "NB0921",
#         "UDS_J",
#         "UDS_H",
#         "UDS_K",
#     ],
# )
