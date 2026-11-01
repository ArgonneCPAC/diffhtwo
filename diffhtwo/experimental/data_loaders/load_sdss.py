from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from diffsky import diffndhist_lomem
from DisCoWebS.data_loader import sdss_loader as sdl
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_transmission_curve

from ..defaults import SDSS_MAGR_THRESH, SDSS_Z_MAX, SDSS_Z_MIN, FilterInfo
from ..latin_hypercube import latin_hypercube as lh
from ..lightcone_generators import generate_lc_data
from .space_utils import (
    AppMagFunc,
    ColorColor,
    CondColor,
    MagColor,
    get_ccd_spaces_at_z,
    get_cmd_spaces_at_z,
    get_condc_spaces_at_z,
    get_mag_spaces_at_z,
    split_space_entries,
)

Sdss = namedtuple(
    "Sdss",
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

# Latin hypercube defaults (currently unused)
LH_N_CENTROIDS = 20_000
LH_SIG = 3.5
LH_D_Z = 0.025

# hole fraction with rmax=0.3 degrees
F_HOLES = 0.003531

# space binning defaults
D_MAG_1d = 0.05
D_MAG_2d = 0.05
GAUSS_SIG_2d = 3.0
D_MAG_COND = 1.0

# Halo lightcone defaults
NUM_HALOS = 150
LGMP_MIN = 10.0
LGMP_MAX = 15.0
LC_SKY_AREA_DEGSQ = 100
N_Z_PHOT_TABLE = 30

SPACES_AT_Z = [
    {
        "z": (0.02, 0.1),
        "md": [
            ("u", True, (0.02, 0.1)),
            ("g", True, (0.02, 0.1)),
            ("r", True, (0.02, 0.1)),
            ("i", True, (0.02, 0.1)),
            ("z", True, (0.02, 0.1)),
        ],
        "ccd": [("Ur_ri", True, (0.02, 0.1)), ("Gr_ri", True, (0.02, 0.1))],
        "cmd": [("R_ur", True, (0.02, 0.1)), ("R_ri", True, (0.02, 0.1))],
        "condc": [
            ("R_ur", True, (0.02, 0.1)),
            ("R_gr", True, (0.02, 0.1)),
            ("R_ri", True, (0.02, 0.1)),
        ],
    },
    {
        "z": (0.1, 0.2),
        "md": [
            ("u", True, (0.1, 0.2)),
            ("g", True, (0.1, 0.2)),
            ("r", True, (0.1, 0.2)),
            ("i", True, (0.1, 0.2)),
            ("z", True, (0.1, 0.2)),
        ],
        "ccd": [("Ur_ri", True, (0.1, 0.2)), ("Gr_ri", True, (0.1, 0.2))],
        "cmd": [("R_ur", True, (0.1, 0.2)), ("R_ri", True, (0.1, 0.2))],
        "condc": [
            ("R_ur", True, (0.1, 0.2)),
            ("R_gr", True, (0.1, 0.2)),
            ("R_ri", True, (0.1, 0.2)),
        ],
    },
]


def apply_ra_dec_cut(
    sdss, ra_min=120, ra_max=240, dec_min=0, dec_max=60, f_holes=F_HOLES
):
    sdss = sdss[
        (sdss["ra"] > ra_min)
        & (sdss["ra"] < ra_max)
        & (sdss["dec"] > dec_min)
        & (sdss["dec"] < dec_max)
    ]
    sky_area_deg2 = compute_sky_area_deg2(ra_min, ra_max, dec_min, dec_max)
    sky_area_deg2 = sky_area_deg2 * (1 - f_holes)
    return sdss, sky_area_deg2


def load_sdss_cuts_applied(drn, sdss_mag_thresh):
    sdss = sdl.load_sdss_cat(drn, apply_cuts=False)

    sdss, sdss_area_degsq = apply_ra_dec_cut(sdss)

    mag_sel_u = (
        (sdss["modelMag_u"] > sdss_mag_thresh.sdss_u[0])
        & (sdss["modelMag_u"] < sdss_mag_thresh.sdss_u[1])
        & (sdss["modelMag_r"] > sdss_mag_thresh.sdss_r[0])
        & (sdss["modelMag_r"] < sdss_mag_thresh.sdss_r[1])
    )

    mag_sel_g = (
        (sdss["modelMag_g"] > sdss_mag_thresh.sdss_g[0])
        & (sdss["modelMag_g"] < sdss_mag_thresh.sdss_g[1])
        & (sdss["modelMag_r"] > sdss_mag_thresh.sdss_r[0])
        & (sdss["modelMag_r"] < sdss_mag_thresh.sdss_r[1])
    )
    mag_sel_r = (sdss["modelMag_r"] > sdss_mag_thresh.sdss_r[0]) & (
        sdss["modelMag_r"] < sdss_mag_thresh.sdss_r[1]
    )
    mag_sel_i = (
        (sdss["modelMag_i"] > sdss_mag_thresh.sdss_i[0])
        & (sdss["modelMag_i"] < sdss_mag_thresh.sdss_i[1])
        & (sdss["modelMag_r"] > sdss_mag_thresh.sdss_r[0])
        & (sdss["modelMag_r"] < sdss_mag_thresh.sdss_r[1])
    )
    mag_sel_z = (
        (sdss["modelMag_z"] > sdss_mag_thresh.sdss_z[0])
        & (sdss["modelMag_z"] < sdss_mag_thresh.sdss_z[1])
        & (sdss["modelMag_r"] > sdss_mag_thresh.sdss_r[0])
        & (sdss["modelMag_r"] < sdss_mag_thresh.sdss_r[1])
    )
    mag_sels = [mag_sel_u, mag_sel_g, mag_sel_r, mag_sel_i, mag_sel_z]
    mag_sels = np.vstack(mag_sels).T

    r_idx = SdssFilters._fields.index("sdss_r")

    return sdss, mag_sels, sdss_area_degsq, r_idx


def compute_sky_area_deg2(ra_min, ra_max, dec_min, dec_max):
    """
    usage: e.g.
    ra_min, ra_max = 120, 240
    dec_min, dec_max = 0, 60
    """

    area_deg2 = (
        (ra_max - ra_min)
        * (np.sin(np.radians(dec_max)) - np.sin(np.radians(dec_min)))
        * (180 / np.pi)
    )
    return area_deg2


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
    spaces_at_z=SPACES_AT_Z,
    d_mag_1d=D_MAG_1d,
    d_mag_2d=D_MAG_2d,
    gauss_sig_2d=GAUSS_SIG_2d,
    d_mag_cond=D_MAG_COND,
    num_halos=NUM_HALOS,
    lgmp_min=LGMP_MIN,
    lgmp_max=LGMP_MAX,
    lc_sky_area_degsq=LC_SKY_AREA_DEGSQ,
    n_z_phot_table=N_Z_PHOT_TABLE,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    sdss_mag_thresh = SdssFilters(
        sdss_u=(15.0, 19.0),
        sdss_g=(13.5, 19.0),
        sdss_r=(13.0, SDSS_MAGR_THRESH),
        sdss_i=(12.5, 17.5),
        sdss_z=(12.5, 17.5),
    )
    sdss, mag_sels, sdss_area_degsq, r_idx = load_sdss_cuts_applied(
        drn, sdss_mag_thresh
    )

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
    redshift = sdss["z"].data

    mags = np.vstack((sdss_u, sdss_g, sdss_r, sdss_i, sdss_z)).T
    mag_labels = [
        r"$u$",
        r"$g$",
        r"$r$",
        r"$i$",
        r"$z$",
    ]

    # derive colors from mags
    sdss_ug = sdss_u - sdss_g
    sdss_gr = sdss_g - sdss_r
    # sdss_ur = sdss_u - sdss_r
    sdss_ri = sdss_r - sdss_i
    sdss_iz = sdss_i - sdss_z

    # stack colors_mag
    dataset = np.vstack(
        (
            sdss_ug,
            sdss_gr,
            sdss_ri,
            sdss_iz,
            sdss_r,
            redshift,
        )
    ).T
    dataset_dim_labels = [
        r"$u - g$",
        r"$g - r$",
        r"$r - i$",
        r"$i - z$",
        r"$r$",
        r"$redshift$",
    ]

    col_idx_lh_dim = [
        [0, 1],  # u - g
        [1, 2],  # g - r
        [2, 3],  # r - i
        [3, 4],  # i - z
    ]
    mag_idx_lh_dim = [2]  # r

    ##############################################################################
    spaces = []
    z_bins = np.array([sp["z"] for sp in spaces_at_z])
    for zbin in range(len(z_bins)):
        z_min, z_max = z_bins[zbin]

        md, md_fit, md_z = split_space_entries(spaces_at_z[zbin].get("md", []))
        ccd, ccd_fit, ccd_z = split_space_entries(spaces_at_z[zbin].get("ccd", []))
        cmd, cmd_fit, cmd_z = split_space_entries(spaces_at_z[zbin].get("cmd", []))
        condc, condc_fit, condc_z = split_space_entries(
            spaces_at_z[zbin].get("condc", [])
        )

        mag_namedtuples = {i: namedtuple(i, AppMagFunc._fields) for i in md}
        ccd_namedtuples = {i: namedtuple(i, ColorColor._fields) for i in ccd}
        cmd_namedtuples = {i: namedtuple(i, MagColor._fields) for i in cmd}
        condc_namedtuples = {i: namedtuple(i, CondColor._fields) for i in condc}

        Spaces = namedtuple(
            "Spaces",
            [
                "z_min",
                "z_max",
                "lc_data",
                *[f"md_{n}" for n in mag_namedtuples],
                *[f"ccd_{n}" for n in ccd_namedtuples],
                *[f"cmd_{n}" for n in cmd_namedtuples],
                *[f"condc_{n}" for n in condc_namedtuples],
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
            SdssFilters,
            md_z,
            md_fit,
            redshift,
            mag_sels,
            mags,
            r_idx,
            d_mag_1d,
            lc_sky_area_degsq,
            data_sky_area_degsq=sdss_area_degsq,
        )

        ccd_z_tuples = get_ccd_spaces_at_z(
            ccd_namedtuples,
            SdssFilters,
            ccd_z,
            ccd_fit,
            redshift,
            mag_sels,
            mags,
            r_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=sdss_area_degsq,
        )

        cmd_z_tuples = get_cmd_spaces_at_z(
            cmd_namedtuples,
            SdssFilters,
            cmd_z,
            cmd_fit,
            redshift,
            mag_sels,
            mags,
            r_idx,
            d_mag_2d,
            gauss_sig_2d,
            lc_sky_area_degsq,
            data_sky_area_degsq=sdss_area_degsq,
        )

        condc_z_lists = get_condc_spaces_at_z(
            condc_namedtuples,
            SdssFilters,
            condc_z,
            condc_fit,
            redshift,
            mag_sels,
            mags,
            r_idx,
            d_mag_1d,
            d_mag_cond,
            lc_sky_area_degsq,
            data_sky_area_degsq=sdss_area_degsq,
        )

        spaces.append(
            Spaces(
                z_min,
                z_max,
                lc_data,
                *mag_z_tuples,
                *ccd_z_tuples,
                *cmd_z_tuples,
                *condc_z_lists,
            )
        )
    ##############################################################################

    return Sdss(
        redshift,
        mags,
        mag_sels,
        mag_labels,
        spaces,
        z_bins,
        filter_info,
        r_idx,
        sdss_area_degsq,
        dataset,
        col_idx_lh_dim,
        mag_idx_lh_dim,
        dataset_dim_labels,
    )


def get_sdss_fitting_data(
    drn,
    ran_key,
    ssp_data,
    spaces_at_z=SPACES_AT_Z,
    d_mag_1d=D_MAG_1d,
    d_mag_2d=D_MAG_2d,
    gauss_sig_2d=GAUSS_SIG_2d,
    d_mag_cond=D_MAG_COND,
    num_halos=NUM_HALOS,
    lgmp_min=LGMP_MIN,
    lgmp_max=LGMP_MAX,
):
    sdss = get_sdss_data(
        drn,
        ran_key,
        ssp_data,
        spaces_at_z=spaces_at_z,
        d_mag_1d=d_mag_1d,
        d_mag_2d=d_mag_2d,
        gauss_sig_2d=gauss_sig_2d,
        d_mag_cond=d_mag_cond,
        num_halos=num_halos,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
    )
    remove = {"dataset_dim_labels", "mags_labels"}
    SdssFitting = namedtuple("Sdss", [f for f in sdss._fields if f not in remove])
    sdss_fitting_data = SdssFitting(
        **{f: getattr(sdss, f) for f in SdssFitting._fields}
    )
    return sdss_fitting_data


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
