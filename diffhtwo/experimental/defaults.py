from collections import namedtuple

from astropy.cosmology import FlatLambdaCDM
from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

# halpha rest wavelength center in fsps
HALPHA_CENTER_AA = 6564.5131

C = 299792458.0  # copied from astropy.constants.c.value in m/s
C_ANGSTROMS = 1e10 * C  # angstrom/s

# astropy's FlatLambdaCDM object for calculations like comoving differential volume
COSMO = FlatLambdaCDM(
    H0=100 * DEFAULT_COSMOLOGY.h,
    Om0=DEFAULT_COSMOLOGY.Om0,
    Ob0=FB * DEFAULT_COSMOLOGY.Om0,
)

FENIKS_AREA_DEG2 = 2828.247933129912 / 3600
FENIKS_Z_MIN = 0.2
FENIKS_Z_MAX = 2.5
FENIKS_MAGK_THRESH = 24.3  # col mag

SDSS_AREA_DEG2 = 7199
SDSS_Z_MIN = 0.02
SDSS_Z_MAX = 0.2
SDSS_MAGR_THRESH = 17.6  # model mag

FilterInfo = namedtuple("FilterInfo", ["mag_thresh", "in_lh", "tcurves"])
DatasetLH = namedtuple(
    "DatasetLH",
    [
        "dataset",
        "dataset_dim_labels",
        "mags",
        "mags_labels",
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

Dataset = namedtuple(
    "Dataset",
    [
        "dataset",
        "dataset_dim_labels",
        "mags",
        "mags_labels",
        "colors",
        "app_mag_funcs",
        "fine_zbins",
        "filter_info",
        "frac_cat",
        "data_sky_area_degsq",
    ],
)

ColorColor = namedtuple(
    "ColorColor", ["col_idx", "sig", "bin_lo", "bin_hi", "N_data", "fit"]
)

ColorCondMag = namedtuple(
    "ColorCondMag",
    [
        "col_idx",
        "cond_idx",
        "cond_min",
        "cond_max",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "fit",
    ],
)

MagColor = namedtuple(
    "MagColor", ["mag_idx", "col_idx", "sig", "bin_lo", "bin_hi", "N_data", "fit"]
)

AppMagFunc = namedtuple(
    "AppMagFunc",
    ["mag_idx", "sig", "bin_lo", "bin_hi", "N_data", "fit"],
)

FeniksFilters = namedtuple(
    "FeniksFilters",
    [
        "MegaCam_uS",
        "HSC_G",
        "HSC_R",
        "HSC_I",
        "NB0816",
        "HSC_Z",
        "NB0921",
        "UDS_J",
        "UDS_H",
        "UDS_K",
    ],
)
