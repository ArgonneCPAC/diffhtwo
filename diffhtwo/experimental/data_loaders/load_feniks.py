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
    AppMagFunc,
    ColorColor,
    ColorCondMag,
    Dataset,
    FilterInfo,
    MagColor,
)
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from ..utils import load_feniks_tcurve
from .N_utils import get_N_1d, get_N_2d

BASE_PATH = Path(__file__).resolve().parent.parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"

Feniks = namedtuple("Feniks", Dataset._fields)

LH_SIG = 3.0
LH_N_CENTROIDS = 60_000

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


def get_feniks_data(
    drn,
    ran_key,
    ssp_data,
    lh_d_mag=0.6,
    phot=PHOT,
    zout=ZOUT,
    num_halos_coarse_zbins=150,
    num_halos_fine_zbins=250,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    mag_bin_edges=None,
):
    # Transmission curves and filter mag thresholds

    feniks_in_lh = FeniksFilters(
        MegaCam_uS=True,
        HSC_G=False,
        HSC_R=False,
        HSC_I=False,
        HSC_Z=False,
        # VIDEO_Y=False,
        UDS_J=False,
        UDS_H=False,
        UDS_K=True,
    )
    tcurves = []
    for feniks_filter in FeniksFilters._fields:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    drn_path = Path(drn)
    phot = ascii.read(drn_path / phot)
    zout = ascii.read(drn_path / zout)

    # get mags
    megacam_uS = get_mag_ab(phot, "fcol_MegaCam_uS")
    hsc_g = get_mag_ab(phot, "fcol_HSC_G")
    hsc_r = get_mag_ab(phot, "fcol_HSC_R")
    hsc_i = get_mag_ab(phot, "fcol_HSC_I")
    hsc_z = get_mag_ab(phot, "fcol_HSC_Z")
    # video_Y = get_mag_ab(phot, "fcol_VIDEO_Y")
    uds_J = get_mag_ab(phot, "fcol_UDS_J")
    uds_H = get_mag_ab(phot, "fcol_UDS_H")
    uds_K = get_mag_ab(phot, "fcol_UDS_K")

    feniks_mag_thresh = FeniksFilters(
        MegaCam_uS=24.9,
        HSC_G=25.1,
        HSC_R=25.3,
        HSC_I=25.1,
        HSC_Z=24.9,
        UDS_J=24.5,
        UDS_H=24.3,
        UDS_K=FENIKS_MAGK_THRESH,
    )

    filter_info = FilterInfo(feniks_mag_thresh, feniks_in_lh, tcurves)

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
    # video_Y = video_Y[mag_thresh]
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
        # & (video_Y != -99)
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
    # video_Y = video_Y[clean]
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
            # video_Y,
            uds_J,
            uds_H,
            uds_K,
            zout["z_phot"],
        )
    ).T

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS - hsc_g
    hsc_gr = hsc_g - hsc_r
    hsc_rz = hsc_r - hsc_z
    hsc_ri = hsc_r - hsc_i
    hsc_iz = hsc_i - hsc_z
    hsc_uds_zJ = hsc_z - uds_J
    uds_JH = uds_J - uds_H
    uds_HK = uds_H - uds_K

    # stack colors_mag
    dataset = np.vstack(
        (
            megacam_hsc_uSg,
            hsc_gr,
            hsc_ri,
            hsc_iz,
            hsc_uds_zJ,
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
        r"$z_{HSC} - J_{UDS}$",
        # r"$Y_{VIDEO} - J_{UDS}$",
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
        # r"$Y_{VIDEO}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
    ]

    # mask redshift
    # z_mask = (zout["z_phot"] > FENIKS_Z_MIN) & (zout["z_phot"] <= FENIKS_Z_MAX)
    # dataset = dataset[z_mask]
    # mags = mags[z_mask]
    # zout = zout[z_mask]

    n_bins = 0

    ##############################################################################
    # prepare 2D and 1D color spaces in coarse z-bins for fitting
    zbins = np.array(
        [
            [0.2, 0.7],
            [0.7, 1.5],
            [1.5, 2.0],
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
    Gr_ri = namedtuple("Gr_ri", ColorColor._fields)
    mag_sel_gr_ri = (
        (hsc_g[z_sel] < feniks_mag_thresh.HSC_G)
        & (hsc_r[z_sel] < feniks_mag_thresh.HSC_R)
        & (hsc_i[z_sel] < feniks_mag_thresh.HSC_I)
    )
    N_gr_ri, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri = get_N_2d(
        hsc_gr[z_sel][mag_sel_gr_ri], hsc_ri[z_sel][mag_sel_gr_ri]
    )
    col_idx = [1, 2, 3]
    gr_ri = Gr_ri(col_idx, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri, N_gr_ri, True)
    n_bins += bin_lo_gr_ri.size

    # 1D (u - g | K)
    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 2)

    ug = []
    Ug_condK = namedtuple("Ug_condK", ColorCondMag._fields)
    mag_sel_ug = (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS) & (
        hsc_g[z_sel] < feniks_mag_thresh.HSC_G
    )
    col_idx = [0, 1]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_ug, sig_ug, bin_lo_ug, bin_hi_ug = get_N_1d(
            megacam_hsc_uSg[z_sel][mag_sel_ug & K_sel]
        )
        ug.append(
            Ug_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_ug,
                bin_lo_ug,
                bin_hi_ug,
                N_1d_ug,
                True,
            )
        )
        n_bins += bin_lo_ug.size

    # 1D (r − i | K)
    ri = []
    Ri_condK = namedtuple("Ri_condK", ColorCondMag._fields)
    mag_sel_ri = (hsc_r[z_sel] < feniks_mag_thresh.HSC_R) & (
        hsc_i[z_sel] < feniks_mag_thresh.HSC_I
    )
    col_idx = [2, 3]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_ri, sig_ri, bin_lo_ri, bin_hi_ri = get_N_1d(
            hsc_ri[z_sel][mag_sel_ri & K_sel]
        )
        ri.append(
            Ri_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_ri,
                bin_lo_ri,
                bin_hi_ri,
                N_1d_ri,
                True,
            )
        )
        n_bins += bin_lo_ri.size

    # 1D (i − z | K)
    iz = []
    Iz_condK = namedtuple("Iz_condK", ColorCondMag._fields)
    mag_sel_iz = (hsc_i[z_sel] < feniks_mag_thresh.HSC_I) & (
        hsc_z[z_sel] < feniks_mag_thresh.HSC_Z
    )
    col_idx = [3, 4]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_iz, sig_iz, bin_lo_iz, bin_hi_iz = get_N_1d(
            hsc_iz[z_sel][mag_sel_iz & K_sel]
        )
        iz.append(
            Iz_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_iz,
                bin_lo_iz,
                bin_hi_iz,
                N_1d_iz,
                True,
            )
        )
        n_bins += bin_lo_iz.size

    # 1D (J − H | K)
    jh = []
    JH_condK = namedtuple("JH_condK", ColorCondMag._fields)
    mag_sel_jh = (uds_J[z_sel] < feniks_mag_thresh.UDS_J) & (
        uds_H[z_sel] < feniks_mag_thresh.UDS_H
    )
    col_idx = [5, 6]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_jh, sig_jh, bin_lo_jh, bin_hi_jh = get_N_1d(
            uds_JH[z_sel][mag_sel_jh & K_sel]
        )
        jh.append(
            JH_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_jh,
                bin_lo_jh,
                bin_hi_jh,
                N_1d_jh,
                True,
            )
        )
        n_bins += bin_lo_jh.size

    # 2D (K, r - i)
    K_ri = namedtuple("K_ri", MagColor._fields)
    mag_sel_ri = (hsc_r[z_sel] < feniks_mag_thresh.HSC_R) & (
        hsc_i[z_sel] < feniks_mag_thresh.HSC_I
    )
    N_K_ri, sig_K_ri, bin_lo_K_ri, bin_hi_K_ri = get_N_2d(
        uds_K[z_sel][mag_sel_ri], hsc_ri[z_sel][mag_sel_ri]
    )
    mag_idx = 7
    col_idx = [2, 3]
    K_ri = K_ri(mag_idx, col_idx, sig_K_ri, bin_lo_K_ri, bin_hi_K_ri, N_K_ri, False)
    n_bins += bin_lo_K_ri.size

    # 2D (K, g - r)
    K_gr = namedtuple("K_gr", MagColor._fields)
    mag_sel_gr = (hsc_g[z_sel] < feniks_mag_thresh.HSC_G) & (
        hsc_r[z_sel] < feniks_mag_thresh.HSC_R
    )
    N_K_gr, sig_K_gr, bin_lo_K_gr, bin_hi_K_gr = get_N_2d(
        uds_K[z_sel][mag_sel_gr], hsc_gr[z_sel][mag_sel_gr]
    )
    mag_idx = 7
    col_idx = [1, 2]
    K_gr = K_gr(mag_idx, col_idx, sig_K_gr, bin_lo_K_gr, bin_hi_K_gr, N_K_gr, False)
    n_bins += bin_lo_K_gr.size

    # 2D (K, J - H)
    K_JH = namedtuple("K_JH", MagColor._fields)
    mag_sel_JH = (uds_J[z_sel] < feniks_mag_thresh.UDS_J) & (
        uds_H[z_sel] < feniks_mag_thresh.UDS_H
    )
    N_K_JH, sig_K_JH, bin_lo_K_JH, bin_hi_K_JH = get_N_2d(
        uds_K[z_sel][mag_sel_JH], uds_JH[z_sel][mag_sel_JH]
    )
    mag_idx = 7
    col_idx = [5, 6]
    K_JH = K_JH(mag_idx, col_idx, sig_K_JH, bin_lo_K_JH, bin_hi_K_JH, N_K_JH, False)
    n_bins += bin_lo_K_JH.size

    z1 = Z1(z_min, z_max, lc_data, gr_ri, ug, ri, iz, jh, K_ri, K_gr, K_JH)
    colors.append(z1)

    ##############################################################################
    # Z2 spaces:
    # 2D (r - z, z - J)
    # 2D (K, u - g)
    # 2D (K, r - z)

    Z2 = namedtuple(
        "Z2",
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
    Rz_zJ = namedtuple("Rz_zJ", ColorColor._fields)
    mag_sel_rz_zJ = (
        (hsc_r[z_sel] < feniks_mag_thresh.HSC_R)
        & (hsc_z[z_sel] < feniks_mag_thresh.HSC_Z)
        & (uds_J[z_sel] < feniks_mag_thresh.UDS_J)
    )
    N_rz_zJ, sig_rz_zJ, bin_lo_rz_zJ, bin_hi_rz_zJ = get_N_2d(
        hsc_rz[z_sel][mag_sel_rz_zJ], hsc_uds_zJ[z_sel][mag_sel_rz_zJ]
    )
    col_idx = [2, 4, 5]
    rz_zJ = Rz_zJ(col_idx, sig_rz_zJ, bin_lo_rz_zJ, bin_hi_rz_zJ, N_rz_zJ, True)
    n_bins += bin_lo_rz_zJ.size

    # 1D (u - g | K)
    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 2)

    ug = []
    mag_sel_ug = (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS) & (
        hsc_g[z_sel] < feniks_mag_thresh.HSC_G
    )
    col_idx = [0, 1]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_ug, sig_ug, bin_lo_ug, bin_hi_ug = get_N_1d(
            megacam_hsc_uSg[z_sel][mag_sel_ug & K_sel]
        )
        ug.append(
            Ug_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_ug,
                bin_lo_ug,
                bin_hi_ug,
                N_1d_ug,
                True,
            )
        )
        n_bins += bin_lo_ug.size

    # 1D (r - z | K)
    rz = []
    Rz_condK = namedtuple("Rz_condK", ColorCondMag._fields)
    mag_sel_rz = (hsc_r[z_sel] < feniks_mag_thresh.HSC_R) & (
        hsc_z[z_sel] < feniks_mag_thresh.HSC_Z
    )
    col_idx = [2, 4]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_rz, sig_rz, bin_lo_rz, bin_hi_rz = get_N_1d(
            hsc_rz[z_sel][mag_sel_rz & K_sel]
        )
        rz.append(
            Rz_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_rz,
                bin_lo_rz,
                bin_hi_rz,
                N_1d_rz,
                True,
            )
        )
        n_bins += bin_lo_rz.size

    # 1D (J − H | K)
    jh = []
    mag_sel_jh = (uds_J[z_sel] < feniks_mag_thresh.UDS_J) & (
        uds_H[z_sel] < feniks_mag_thresh.UDS_H
    )
    col_idx = [5, 6]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_jh, sig_jh, bin_lo_jh, bin_hi_jh = get_N_1d(
            uds_JH[z_sel][mag_sel_jh & K_sel]
        )
        jh.append(
            JH_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_jh,
                bin_lo_jh,
                bin_hi_jh,
                N_1d_jh,
                True,
            )
        )
        n_bins += bin_lo_jh.size

    # 2D (K, u - g)
    K_ug = namedtuple("K_ug", MagColor._fields)
    mag_sel_ug = (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS) & (
        hsc_g[z_sel] < feniks_mag_thresh.HSC_G
    )
    N_K_ug, sig_K_ug, bin_lo_K_ug, bin_hi_K_ug = get_N_2d(
        uds_K[z_sel][mag_sel_ug], megacam_hsc_uSg[z_sel][mag_sel_ug]
    )
    mag_idx = 7
    col_idx = [0, 1]
    K_ug = K_ug(mag_idx, col_idx, sig_K_ug, bin_lo_K_ug, bin_hi_K_ug, N_K_ug, False)
    n_bins += bin_lo_K_ug.size

    # 2D (K, r - z)
    K_rz = namedtuple("K_rz", MagColor._fields)
    mag_sel_rz = (hsc_r[z_sel] < feniks_mag_thresh.HSC_R) & (
        hsc_z[z_sel] < feniks_mag_thresh.HSC_Z
    )
    N_K_rz, sig_K_rz, bin_lo_K_rz, bin_hi_K_rz = get_N_2d(
        uds_K[z_sel][mag_sel_rz], hsc_rz[z_sel][mag_sel_rz]
    )
    mag_idx = 7
    col_idx = [2, 4]
    K_rz = K_rz(mag_idx, col_idx, sig_K_rz, bin_lo_K_rz, bin_hi_K_rz, N_K_rz, False)
    n_bins += bin_lo_K_rz.size

    z2 = Z2(z_min, z_max, lc_data, rz_zJ, ug, rz, jh, K_ug, K_rz)
    colors.append(z2)

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
    zJ_JH = namedtuple("zJ_JH", ColorColor._fields)
    mag_sel_zJ_JH = (
        (hsc_z[z_sel] < feniks_mag_thresh.HSC_Z)
        & (uds_J[z_sel] < feniks_mag_thresh.UDS_J)
        & (uds_H[z_sel] < feniks_mag_thresh.UDS_H)
    )
    N_zJ_JH, sig_zJ_JH, bin_lo_zJ_JH, bin_hi_zJ_JH = get_N_2d(
        hsc_uds_zJ[z_sel][mag_sel_zJ_JH], uds_JH[z_sel][mag_sel_zJ_JH]
    )
    col_idx = [4, 5, 6]
    zJ_JH = zJ_JH(col_idx, sig_zJ_JH, bin_lo_zJ_JH, bin_hi_zJ_JH, N_zJ_JH, True)
    n_bins += bin_lo_zJ_JH.size

    # 2D (u - g, g - r)
    Ug_gr = namedtuple("Ug_gr", ColorColor._fields)
    mag_sel_ugr = (
        (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS)
        & (hsc_g[z_sel] < feniks_mag_thresh.HSC_G)
        & (hsc_r[z_sel] < feniks_mag_thresh.HSC_R)
    )
    N_ug_gr, sig_ug_gr, bin_lo_ug_gr, bin_hi_ug_gr = get_N_2d(
        megacam_hsc_uSg[z_sel][mag_sel_ugr], hsc_gr[z_sel][mag_sel_ugr]
    )
    col_idx = [0, 1, 2]
    ug_gr = Ug_gr(col_idx, sig_ug_gr, bin_lo_ug_gr, bin_hi_ug_gr, N_ug_gr, True)
    n_bins += bin_lo_ug_gr.size

    # 1D (u - g | K)
    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 4)

    ug = []
    mag_sel_ug = (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS) & (
        hsc_g[z_sel] < feniks_mag_thresh.HSC_G
    )
    col_idx = [0, 1]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_ug, sig_ug, bin_lo_ug, bin_hi_ug = get_N_1d(
            megacam_hsc_uSg[z_sel][mag_sel_ug & K_sel]
        )
        ug.append(
            Ug_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_ug,
                bin_lo_ug,
                bin_hi_ug,
                N_1d_ug,
                True,
            )
        )
        n_bins += bin_lo_ug.size

    # 1D (g - r | K)
    gr = []
    Gr_condK = namedtuple("Gr_condK", ColorCondMag._fields)
    mag_sel_gr = (hsc_g[z_sel] < feniks_mag_thresh.HSC_G) & (
        hsc_r[z_sel] < feniks_mag_thresh.HSC_R
    )
    col_idx = [1, 2]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_gr, sig_gr, bin_lo_gr, bin_hi_gr = get_N_1d(
            hsc_gr[z_sel][mag_sel_gr & K_sel]
        )
        gr.append(
            Gr_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_gr,
                bin_lo_gr,
                bin_hi_gr,
                N_1d_gr,
                True,
            )
        )
        n_bins += bin_lo_gr.size

    # 1D (J − H | K)
    jh = []
    mag_sel_jh = (uds_J[z_sel] < feniks_mag_thresh.UDS_J) & (
        uds_H[z_sel] < feniks_mag_thresh.UDS_H
    )
    col_idx = [5, 6]
    cond_idx = 7
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d_jh, sig_jh, bin_lo_jh, bin_hi_jh = get_N_1d(
            uds_JH[z_sel][mag_sel_jh & K_sel]
        )
        jh.append(
            JH_condK(
                col_idx,
                cond_idx,
                Kbins[k],
                Kbins[k + 1],
                sig_jh,
                bin_lo_jh,
                bin_hi_jh,
                N_1d_jh,
                True,
            )
        )
        n_bins += bin_lo_jh.size

    # 2D (K, u - g)
    K_ug = namedtuple("K_ug", MagColor._fields)
    mag_sel_ug = (megacam_uS[z_sel] < feniks_mag_thresh.MegaCam_uS) & (
        hsc_g[z_sel] < feniks_mag_thresh.HSC_G
    )
    N_K_ug, sig_K_ug, bin_lo_K_ug, bin_hi_K_ug = get_N_2d(
        uds_K[z_sel][mag_sel_ug], megacam_hsc_uSg[z_sel][mag_sel_ug]
    )
    mag_idx = 7
    col_idx = [0, 1]
    K_ug = K_ug(mag_idx, col_idx, sig_K_ug, bin_lo_K_ug, bin_hi_K_ug, N_K_ug, False)
    n_bins += bin_lo_K_ug.size

    # 2D (K, g - r)
    K_gr = namedtuple("K_gr", MagColor._fields)
    mag_sel_gr = (hsc_g[z_sel] < feniks_mag_thresh.HSC_G) & (
        hsc_r[z_sel] < feniks_mag_thresh.HSC_R
    )
    N_K_gr, sig_K_gr, bin_lo_K_gr, bin_hi_K_gr = get_N_2d(
        uds_K[z_sel][mag_sel_gr], hsc_gr[z_sel][mag_sel_gr]
    )
    mag_idx = 7
    col_idx = [1, 2]
    K_gr = K_gr(mag_idx, col_idx, sig_K_gr, bin_lo_K_gr, bin_hi_K_gr, N_K_gr, False)
    n_bins += bin_lo_K_gr.size

    # 2D (K, J - H)
    K_JH = namedtuple("K_JH", MagColor._fields)
    mag_sel_JH = (uds_J[z_sel] < feniks_mag_thresh.UDS_J) & (
        uds_H[z_sel] < feniks_mag_thresh.UDS_H
    )
    N_K_JH, sig_K_JH, bin_lo_K_JH, bin_hi_K_JH = get_N_2d(
        uds_K[z_sel][mag_sel_JH], uds_JH[z_sel][mag_sel_JH]
    )
    mag_idx = 7
    col_idx = [5, 6]
    K_JH = K_JH(mag_idx, col_idx, sig_K_JH, bin_lo_K_JH, bin_hi_K_JH, N_K_JH, False)
    n_bins += bin_lo_K_JH.size

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
        ]
    )
    ##############################################################################
    AppMagFuncs = namedtuple(
        "AppMagFuncs",
        ["z_min", "z_max", "lc_data", "u", "g", "r", "i", "z", "J", "H", "K"],
    )

    U = namedtuple("U", AppMagFunc._fields)
    G = namedtuple("G", AppMagFunc._fields)
    R = namedtuple("R", AppMagFunc._fields)
    I = namedtuple("I", AppMagFunc._fields)
    Z = namedtuple("Z", AppMagFunc._fields)
    J = namedtuple("J", AppMagFunc._fields)
    H = namedtuple("H", AppMagFunc._fields)
    K = namedtuple("K", AppMagFunc._fields)

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
        mag_idx_u = 0
        N_1d_u, sig_u, bin_lo_u, bin_hi_u = get_N_1d(megacam_uS[z_sel])
        u = U(mag_idx_u, sig_u, bin_lo_u, bin_hi_u, N_1d_u, True)
        n_bins += bin_lo_u.size

        # 1D (g)
        mag_idx_g = 1
        N_1d_g, sig_g, bin_lo_g, bin_hi_g = get_N_1d(hsc_g[z_sel])
        g = G(mag_idx_g, sig_g, bin_lo_g, bin_hi_g, N_1d_g, False)
        n_bins += bin_lo_g.size

        # 1D (r)
        mag_idx_r = 2
        N_1d_r, sig_r, bin_lo_r, bin_hi_r = get_N_1d(hsc_r[z_sel])
        r = R(mag_idx_r, sig_r, bin_lo_r, bin_hi_r, N_1d_r, True)
        n_bins += bin_lo_r.size

        # 1D (i)
        mag_idx_i = 3
        N_1d_i, sig_i, bin_lo_i, bin_hi_i = get_N_1d(hsc_i[z_sel])
        i = I(mag_idx_i, sig_i, bin_lo_i, bin_hi_i, N_1d_i, False)
        n_bins += bin_lo_i.size

        # 1D (z)
        mag_idx_z = 4
        N_1d_z, sig_z, bin_lo_z, bin_hi_z = get_N_1d(hsc_z[z_sel])
        z = Z(mag_idx_z, sig_z, bin_lo_z, bin_hi_z, N_1d_z, False)
        n_bins += bin_lo_z.size

        # 1D (J)
        mag_idx_j = 5
        N_1d_j, sig_j, bin_lo_j, bin_hi_j = get_N_1d(uds_J[z_sel])
        j = J(mag_idx_j, sig_j, bin_lo_j, bin_hi_j, N_1d_j, False)
        n_bins += bin_lo_j.size

        # 1D (H)
        mag_idx_h = 6
        N_1d_h, sig_h, bin_lo_h, bin_hi_h = get_N_1d(uds_H[z_sel])
        h = H(mag_idx_h, sig_h, bin_lo_h, bin_hi_h, N_1d_h, False)
        n_bins += bin_lo_h.size

        # 1D (K)
        mag_idx_k = 7
        N_1d_k, sig_k, bin_lo_k, bin_hi_k = get_N_1d(uds_K[z_sel])
        k = K(mag_idx_k, sig_k, bin_lo_k, bin_hi_k, N_1d_k, True)
        n_bins += bin_lo_k.size

        app_mag_funcs.append(AppMagFuncs(z_min, z_max, lc_data, u, g, r, i, z, j, h, k))

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

    return Feniks(
        dataset,
        dataset_dim_labels,
        mags,
        mags_labels,
        colors,
        app_mag_funcs,
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
        "UDS_J",
        "UDS_H",
        "UDS_K",
    ],
)
