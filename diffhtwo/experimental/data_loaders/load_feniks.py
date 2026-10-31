import warnings
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import ascii
from diffsky import diffndhist_lomem
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders.defaults import TransmissionCurve
from scipy import optimize

from ..defaults import (
    FENIKS_AREA_DEG2,
    FENIKS_MAGK_THRESH,
    FENIKS_Z_MAX,
    FENIKS_Z_MIN,
    FilterInfo,
)
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from ..utils import add_random_rows, load_feniks_tcurve
from .space_utils import (
    AppMagFunc,
    ColorColor,
    MagColor,
    get_ccd_spaces_at_z,
    get_cmd_spaces_at_z,
    get_mag_spaces_at_z,
    split_space_entries,
)

BASE_PATH = Path(__file__).resolve().parent.parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


PHOT = "feniks_phot_selected.cat"
ZOUT = "feniks_zout_selected.ecsv"

Feniks = namedtuple(
    "Feniks",
    [
        "redshift",
        "mags",
        "mag_sels",
        "mags_labels",
        "spaces",
        "zbins",
        "filter_info",
        "parent_cut_idx",
        "data_sky_area_degsq",
        "dataset",
        "col_idx",
        "mag_idx",
        "dataset_dim_labels",
    ],
)

LH_SIG = 3.0
LH_N_CENTROIDS = 30_000
LH_D_Z = 0.3

D_MAG_1d = 0.2
D_MAG_2d = 0.2
GAUSS_SIG_2d = 3.0
zbins = np.array(
    [
        [0.4, 0.7],
        [0.7, 1.0],
        [1.0, 1.5],
        [1.5, 2.0],
    ]
)
SPACES_AT_Z = [
    {
        "z": (0.4, 0.7),
        "md": [
            ("U", True, (0.4, 0.7)),
            ("G", True, (0.4, 0.7)),
            ("R", True, (0.4, 0.7)),
            ("I", True, (0.4, 0.7)),
            ("Z", True, (0.4, 0.7)),
            ("J", True, (0.4, 0.7)),
            ("H", True, (0.4, 0.7)),
            ("K", True, (0.4, 0.7)),
        ],
        "ccd": [("Gr_ri", True, (0.4, 0.7))],
        "cmd": [
            ("K_ri", True, (0.4, 0.7)),
            ("K_gr", True, (0.4, 0.7)),
            ("K_JH", True, (0.4, 0.7)),
        ],
    },
    {
        "z": (0.7, 1.0),
        "md": [
            ("U", True, (0.7, 1.0)),
            ("G", True, (0.7, 1.0)),
            ("R", True, (0.7, 1.0)),
            ("I", True, (0.7, 1.0)),
            ("Z", True, (0.7, 1.0)),
            ("J", True, (0.7, 1.0)),
            ("H", True, (0.7, 1.0)),
            ("K", True, (0.7, 1.0)),
        ],
        "ccd": [("rz_zJ", True, (0.7, 1.0))],
        "cmd": [
            ("K_ug", True, (0.7, 1.0)),
            ("K_rz", True, (0.7, 1.0)),
            ("K_JH", True, (0.7, 1.0)),
        ],
    },
    {
        "z": (1.0, 1.5),
        "md": [
            ("U", True, (1.0, 1.5)),
            ("G", True, (1.0, 1.5)),
            ("R", True, (1.0, 1.5)),
            ("I", True, (1.0, 1.5)),
            ("Z", True, (1.0, 1.5)),
            ("J", True, (1.0, 1.5)),
            ("H", True, (1.0, 1.5)),
            ("K", True, (1.0, 1.5)),
        ],
        "ccd": [("rz_zJ", True, (1.0, 1.5))],
        "cmd": [
            ("K_ug", True, (1.0, 1.5)),
            ("K_rz", True, (1.0, 1.5)),
            ("K_JH", True, (1.0, 1.5)),
        ],
    },
    {
        "z": (1.5, 2.0),
        "md": [
            ("U", True, (1.5, 2.0)),
            ("G", True, (1.5, 2.0)),
            ("R", True, (1.5, 2.0)),
            ("I", True, (1.5, 2.0)),
            ("Z", True, (1.5, 2.0)),
            ("J", True, (1.5, 2.0)),
            ("H", True, (1.5, 2.0)),
            ("K", True, (1.5, 2.0)),
        ],
        "ccd": [("ZJ_JH", True, (1.5, 2.0)), ("Ug_gr", True, (1.5, 2.0))],
        "cmd": [
            ("K_ug", True, (1.5, 2.0)),
            ("K_gr", True, (1.5, 2.0)),
            ("K_JH", True, (1.5, 2.0)),
        ],
    },
]


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


def get_mag_ab_col(phot_table, col_name, ZP=25.0):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_ab_col = -2.5 * np.log10(phot_table[col_name]) + ZP

    mag_ab_col[~np.isfinite(mag_ab_col)] = -99.0
    mag_ab_col = mag_ab_col.data

    return mag_ab_col


def get_mag_ab_tot(phot_table, col_name, ZP=25.0):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mag_ab_tot = (
            -2.5 * np.log10(phot_table[col_name] * phot_table["aper_to_tot_corr"]) + ZP
        )

    mag_ab_tot[~np.isfinite(mag_ab_tot)] = -99.0
    mag_ab_tot = mag_ab_tot.data

    return mag_ab_tot


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
    mu[-3] = mu[-3] - 1.0  # u
    mu[-2] = mu[-2] - 1.0  # K

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
    spaces_at_z=SPACES_AT_Z,
    d_mag_1d=D_MAG_1d,
    d_mag_2d=D_MAG_2d,
    gauss_sig_2d=GAUSS_SIG_2d,
    num_halos=100,
    phot=PHOT,
    zout=ZOUT,
    lgmp_min=10.0,
    lgmp_max=15.0,
    lc_sky_area_degsq=100,
    n_z_phot_table=30,
    add_random_rows_for_testing=False,
    testing=False,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    drn_path = Path(drn)
    phot = ascii.read(drn_path / phot)
    zout = ascii.read(drn_path / zout)

    if add_random_rows_for_testing:
        phot = add_random_rows(phot, N=400)
        zout = add_random_rows(zout, N=400)

    # get total and optimal aperture (for colors) mags
    megacam_uS_col = get_mag_ab_col(phot, "fcol_MegaCam_uS")
    megacam_uS_tot = get_mag_ab_tot(phot, "fcol_MegaCam_uS")

    hsc_g_col = get_mag_ab_col(phot, "fcol_HSC_G")
    hsc_g_tot = get_mag_ab_tot(phot, "fcol_HSC_G")

    hsc_r_col = get_mag_ab_col(phot, "fcol_HSC_R")
    hsc_r_tot = get_mag_ab_tot(phot, "fcol_HSC_R")

    hsc_i_col = get_mag_ab_col(phot, "fcol_HSC_I")
    hsc_i_tot = get_mag_ab_tot(phot, "fcol_HSC_I")

    hsc_z_col = get_mag_ab_col(phot, "fcol_HSC_Z")
    hsc_z_tot = get_mag_ab_tot(phot, "fcol_HSC_Z")

    uds_J_col = get_mag_ab_col(phot, "fcol_UDS_J")
    uds_J_tot = get_mag_ab_tot(phot, "fcol_UDS_J")

    uds_H_col = get_mag_ab_col(phot, "fcol_UDS_H")
    uds_H_tot = get_mag_ab_tot(phot, "fcol_UDS_H")

    uds_K_col = get_mag_ab_col(phot, "fcol_UDS_K")
    uds_K_tot = get_mag_ab_tot(phot, "fcol_UDS_K")

    """
    The FENIKS_AREA_DEG2 is calculated using individual band masks which were used to assign
    flux in a given band to -99.0. for objects near bright stars, etc.
    This means that the removal of objects as below will be mostly from the masked areas already taken into account in area calculation.
    That's good. However, the masks sets flux=-99.0 only for bad areas. What about objects with flux<0 & flux!=-99.0, essentially the droputs?
    These objects will get their magnitudes be set to -99.0 through the get_mag_ab functions above. So, they will
    also be removed below. It turns out that's ok! We don't need to rederive area due to dropout removal because, these dropouts will anyway be discarded
    due to the magnitude cuts defined in feniks_mag_thresh
    """
    clean = (
        (megacam_uS_col != -99.0)
        & (megacam_uS_tot != -99.0)
        & (hsc_g_col != -99.0)
        & (hsc_g_tot != -99.0)
        & (hsc_r_col != -99.0)
        & (hsc_r_tot != -99.0)
        & (hsc_i_col != -99.0)
        & (hsc_i_tot != -99.0)
        & (hsc_z_col != -99.0)
        & (hsc_z_tot != -99.0)
        & (uds_J_col != -99.0)
        & (uds_J_tot != -99.0)
        & (uds_H_col != -99.0)
        & (uds_H_tot != -99.0)
        & (uds_K_col != -99.0)
        & (uds_K_tot != -99.0)
    )

    phot = phot[clean]
    zout = zout[clean]
    z_best = zout["z_phot"].data

    megacam_uS_col = megacam_uS_col[clean]
    megacam_uS_tot = megacam_uS_tot[clean]

    hsc_g_col = hsc_g_col[clean]
    hsc_g_tot = hsc_g_tot[clean]

    hsc_r_col = hsc_r_col[clean]
    hsc_r_tot = hsc_r_tot[clean]

    hsc_i_col = hsc_i_col[clean]
    hsc_i_tot = hsc_i_tot[clean]

    hsc_z_col = hsc_z_col[clean]
    hsc_z_tot = hsc_z_tot[clean]

    uds_J_col = uds_J_col[clean]
    uds_J_tot = uds_J_tot[clean]

    uds_H_col = uds_H_col[clean]
    uds_H_tot = uds_H_tot[clean]

    uds_K_col = uds_K_col[clean]
    uds_K_tot = uds_K_tot[clean]

    mags = np.vstack(
        (
            megacam_uS_tot,
            hsc_g_tot,
            hsc_r_tot,
            hsc_i_tot,
            hsc_z_tot,
            uds_J_tot,
            uds_H_tot,
            uds_K_tot,
        )
    ).T

    mag_labels = [
        r"$uS_{MegaCam}$",
        r"$g_{HSC}$",
        r"$r_{HSC}$",
        r"$i_{HSC}$",
        r"$z_{HSC}$",
        r"$J_{UDS}$",
        r"$H_{UDS}$",
        r"$K_{UDS}$",
    ]

    feniks_mag_thresh = FeniksFilters(
        MegaCam_uS=(21.4, 25.0),
        HSC_G=(20.6, 25.3),
        HSC_R=(19.8, 24.8),
        HSC_I=(19.0, 24.5),
        HSC_Z=(18.8, 24.3),
        UDS_J=(18.0, 24.0),
        UDS_H=(17.5, 24.0),
        UDS_K=(17.0, FENIKS_MAGK_THRESH),
    )

    # Transmission curves and filter mag thresholds
    tcurves = []
    mag_sel_per_band = []

    K_mag_thresh = getattr(feniks_mag_thresh, "UDS_K")
    K_idx = FeniksFilters._fields.index("UDS_K")
    K_mag_sel = (mags[:, K_idx] > K_mag_thresh[0]) & (mags[:, K_idx] < K_mag_thresh[1])
    for f in range(len(FeniksFilters._fields)):
        feniks_filter = FeniksFilters._fields[f]

        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

        mag_limit = getattr(feniks_mag_thresh, feniks_filter)
        mag_sel = (mags[:, f] > mag_limit[0]) & (mags[:, f] < mag_limit[1])
        mag_sel *= K_mag_sel

        mag_sel_per_band.append(mag_sel)

    mag_sels = np.vstack(mag_sel_per_band).T
    filter_info = FilterInfo(feniks_mag_thresh, tcurves)

    # derive colors from mags
    megacam_hsc_uSg = megacam_uS_col - hsc_g_col
    hsc_gr = hsc_g_col - hsc_r_col
    hsc_ri = hsc_r_col - hsc_i_col
    hsc_iz = hsc_i_col - hsc_z_col
    hsc_rz = hsc_r_col - hsc_z_col
    hsc_uds_zJ = hsc_z_col - uds_J_col
    uds_JH = uds_J_col - uds_H_col
    uds_HK = uds_H_col - uds_K_col

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
            megacam_uS_tot,
            uds_K_tot,
            z_best,
        )
    ).T

    col_idx_lh_dim = [
        [0, 1],  # u - g
        [1, 2],  # g - r
        [2, 3],  # r - i
        [3, 4],  # i - z
        [4, 5],  # z - J
        [5, 6],  # J - H
        [6, 7],  # H - K
    ]
    mag_idx_lh_dim = [
        0,  # u
        7,  # K
    ]
    dataset_dim_labels = [
        r"$u - g$",
        r"$g - r$",
        r"$r - i$",
        r"$i - z$",
        r"$z - J$",
        r"$J - H$",
        r"$H - K$",
        r"$uS$",
        r"$K$",
        r"$redshift$",
    ]

    ##############################################################################
    spaces = []
    z_bins = np.array([sp["z"] for sp in spaces_at_z])
    for zbin in range(len(z_bins)):
        z_min, z_max = z_bins[zbin]

        md, md_fit, md_z = split_space_entries(spaces_at_z[zbin].get("md", []))
        ccd, ccd_fit, ccd_z = split_space_entries(spaces_at_z[zbin].get("ccd", []))
        cmd, cmd_fit, cmd_z = split_space_entries(spaces_at_z[zbin].get("cmd", []))

        mag_namedtuples = {i: namedtuple(i, AppMagFunc._fields) for i in md}
        ccd_namedtuples = {i: namedtuple(i, ColorColor._fields) for i in ccd}
        cmd_namedtuples = {i: namedtuple(i, MagColor._fields) for i in cmd}

        Spaces = namedtuple(
            "Spaces",
            [
                "z_min",
                "z_max",
                "lc_data",
                *mag_namedtuples,
                *ccd_namedtuples,
                *cmd_namedtuples,
            ],
        )

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

        lc_data = generate_lc_data(*lc_args)

        mag_z_tuples = get_mag_spaces_at_z(
            mag_namedtuples,
            FeniksFilters,
            md_z,
            md_fit,
            z_best,
            mag_sels,
            mags,
            K_idx,
            d_mag_1d,
            lc_sky_area_degsq,
            data_sky_area_degsq=FENIKS_AREA_DEG2,
        )

        ccd_z_tuples = get_ccd_spaces_at_z(
            ccd_namedtuples,
            FeniksFilters,
            ccd_z,
            ccd_fit,
            z_best,
            mag_sels,
            mags,
            K_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=FENIKS_AREA_DEG2,
        )

        cmd_z_tuples = get_cmd_spaces_at_z(
            cmd_namedtuples,
            FeniksFilters,
            cmd_z,
            cmd_fit,
            z_best,
            mag_sels,
            mags,
            K_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=FENIKS_AREA_DEG2,
        )

        spaces.append(
            Spaces(
                z_min,
                z_max,
                lc_data,
                *mag_z_tuples,
                *ccd_z_tuples,
                *cmd_z_tuples,
            )
        )

    return Feniks(
        z_best,
        mags,
        mag_sels,
        mag_labels,
        spaces,
        zbins,
        filter_info,
        K_idx,
        FENIKS_AREA_DEG2,
        dataset,
        col_idx_lh_dim,
        mag_idx_lh_dim,
        dataset_dim_labels,
    )


def get_feniks_fitting_data(
    feniks_drn,
    ran_key,
    ssp_data,
    num_halos=100,
    phot=PHOT,
    zout=ZOUT,
    lgmp_min=10.0,
    lgmp_max=15.0,
    add_random_rows_for_testing=False,
    testing=False,
):
    feniks = get_feniks_data(
        feniks_drn,
        ran_key,
        ssp_data,
        num_halos=num_halos,
        phot=phot,
        zout=zout,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        add_random_rows_for_testing=add_random_rows_for_testing,
    )
    remove = {"dataset_dim_labels", "mags_labels"}
    FeniksFitting = namedtuple("Feniks", [f for f in feniks._fields if f not in remove])
    feniks_fitting_data = FeniksFitting(
        **{f: getattr(feniks, f) for f in FeniksFitting._fields}
    )

    return feniks_fitting_data


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
