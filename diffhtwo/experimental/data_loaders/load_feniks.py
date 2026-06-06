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
    FilterInfo,
)
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from ..utils import load_feniks_tcurve

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


def get_N_1d(dim1, dim1_bin_edges=None, dmag=0.2, sig_scale=0.5):
    dataset = dim1.reshape(dim1.size, 1)
    if dim1_bin_edges is None:
        dim1_bin_edges = np.arange(dim1.min(), dim1.max(), dmag)

    bin_lo = dim1_bin_edges[:-1].reshape(dim1_bin_edges[:-1].size, 1)
    bin_hi = dim1_bin_edges[1:].reshape(dim1_bin_edges[1:].size, 1)

    sig = jnp.zeros_like(bin_lo) + (dmag * sig_scale)

    N_1d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return (
        N_1d,
        sig,
        bin_lo,
        bin_hi,
    )


def get_N_2d(dim1, dim2, sig_scale=0.5):
    dataset = np.vstack((dim1, dim2)).T

    dim1_bin_edges = np.linspace(dim1.min(), dim1.max(), 11)
    dim2_bin_edges = np.linspace(dim2.min(), dim2.max(), 11)

    dim1_lo = dim1_bin_edges[:-1]
    dim2_lo = dim2_bin_edges[:-1]
    bin_lo = np.meshgrid(dim1_lo, dim2_lo, indexing="ij")
    bin_lo = np.array(bin_lo).T.reshape(-1, 2)

    dim1_hi = dim1_bin_edges[1:]
    dim2_hi = dim2_bin_edges[1:]
    bin_hi = np.meshgrid(dim1_hi, dim2_hi, indexing="ij")
    bin_hi = np.array(bin_hi).T.reshape(-1, 2)

    sig1 = np.diff(dim1_bin_edges) * sig_scale
    sig2 = np.diff(dim2_bin_edges) * sig_scale
    sig = np.meshgrid(sig1, sig2, indexing="ij")
    sig = np.array(sig).T.reshape(-1, 2)

    N_2d = diffndhist_lomem.tw_ndhist(
        dataset,
        sig,
        bin_lo,
        bin_hi,
    )

    return N_2d, sig, bin_lo, bin_hi


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
    num_halos=250,
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
        # & (video_Y < feniks_mag_thresh.VIDEO_Y)
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

    N_obj_pre_cuts = len(zout)

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

    N_obj_post_cuts = len(zout)
    frac_cat = N_obj_post_cuts / N_obj_pre_cuts

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
    # video_uds_YJ = video_Y - uds_J
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

    # prepare 1D app mag functions in z-bins for fitting

    zbins = np.array(
        [
            [0.2, 0.7],
            [0.7, 1.5],
            [1.5, 2.5],
        ]
    )

    lc_data = []

    # Z1 --> get spaces: 2D (g-r, r-i), 1D (u-g | K)
    zbin = 0
    z_min = zbins[zbin][0]
    z_max = zbins[zbin][1]

    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )
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

    lc_data.append(generate_lc_data(*lc_args))

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)
    Z1 = namedtuple(
        "Z1",
        [
            "z_min",
            "z_max",
            "gr_ri",
            "ug",
        ],
    )

    Gr_ri = namedtuple("Gr_ri", ["N", "sig", "bin_lo", "bin_hi"])
    N_gr_ri, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri = get_N_2d(
        hsc_gr[z_sel], hsc_ri[z_sel]
    )
    gr_ri = Gr_ri(N_gr_ri, sig_gr_ri, bin_lo_gr_ri, bin_hi_gr_ri)

    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 2)
    ug = []
    Ug_condK = namedtuple(
        "Ug_condK", ["K_min", "K_max", "N", "sig", "bin_lo", "bin_hi"]
    )
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d, sig, bin_lo, bin_hi = get_N_1d(megacam_hsc_uSg[z_sel][K_sel])
        ug.append(Ug_condK(Kbins[k], Kbins[k + 1], N_1d, sig, bin_lo, bin_hi))

    z1 = Z1(z_min, z_max, gr_ri, ug)

    # Z2 --> get spaces: 2D (r-z, z-J), 1D (u-g | K)
    zbin = 1
    z_min = zbins[zbin][0]
    z_max = zbins[zbin][1]

    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )
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

    lc_data.append(generate_lc_data(*lc_args))

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)
    Z2 = namedtuple(
        "Z2",
        [
            "z_min",
            "z_max",
            "rz_zJ",
            "ug",
        ],
    )

    Rz_zJ = namedtuple("Rz_zJ", ["N", "sig", "bin_lo", "bin_hi"])
    N_rz_zJ, sig_rz_zJ, bin_lo_rz_zJ, bin_hi_rz_zJ = get_N_2d(
        hsc_rz[z_sel], hsc_uds_zJ[z_sel]
    )
    rz_zJ = Rz_zJ(N_rz_zJ, sig_rz_zJ, bin_lo_rz_zJ, bin_hi_rz_zJ)

    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 2)
    ug = []
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d, sig, bin_lo, bin_hi = get_N_1d(megacam_hsc_uSg[z_sel][K_sel])
        ug.append(Ug_condK(Kbins[k], Kbins[k + 1], N_1d, sig, bin_lo, bin_hi))

    z2 = Z2(z_min, z_max, rz_zJ, ug)

    # Z3 --> get spaces: 2D (z-J, J-H), 1D (u-g | K), 1D (g-r | K)
    zbin = 2
    z_min = zbins[zbin][0]
    z_max = zbins[zbin][1]

    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )
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

    lc_data.append(generate_lc_data(*lc_args))

    z_sel = (zout["z_phot"] > z_min) & (zout["z_phot"] <= z_max)
    Z3 = namedtuple(
        "Z3",
        ["z_min", "z_max", "zJ_JH", "ug", "gr"],
    )

    zJ_JH = namedtuple("zJ_JH", ["N", "sig", "bin_lo", "bin_hi"])
    N_zJ_JH, sig_zJ_JH, bin_lo_zJ_JH, bin_hi_zJ_JH = get_N_2d(
        hsc_uds_zJ[z_sel], uds_JH[z_sel]
    )
    zJ_JH = zJ_JH(N_zJ_JH, sig_zJ_JH, bin_lo_zJ_JH, bin_hi_zJ_JH)

    Kbins = np.arange(uds_K[z_sel].min(), uds_K[z_sel].max(), 4)
    ug = []
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d, sig, bin_lo, bin_hi = get_N_1d(megacam_hsc_uSg[z_sel][K_sel])
        ug.append(Ug_condK(Kbins[k], Kbins[k + 1], N_1d, sig, bin_lo, bin_hi))

    gr = []
    Gr_condK = namedtuple(
        "Gr_condK", ["K_min", "K_max", "N", "sig", "bin_lo", "bin_hi"]
    )
    for k in range(len(Kbins) - 1):
        K_sel = (uds_K[z_sel] > Kbins[k]) & (uds_K[z_sel] <= Kbins[k + 1])
        N_1d, sig, bin_lo, bin_hi = get_N_1d(hsc_gr[z_sel][K_sel])
        gr.append(Gr_condK(Kbins[k], Kbins[k + 1], N_1d, sig, bin_lo, bin_hi))

    z3 = Z3(z_min, z_max, zJ_JH, ug, gr)

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
        z1,
        z2,
        z3,
        lc_data,
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
        # "VIDEO_Y",
        "UDS_J",
        "UDS_H",
        "UDS_K",
    ],
)
