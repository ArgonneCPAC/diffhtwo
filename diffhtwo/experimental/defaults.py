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

# FENIKS_AREA_DEG2 is Area with combined coverage in the following bands:
# ["MegaCam_uS", HSC_G", "HSC_R", "HSC_I", "HSC_Z", "UDS_J", "UDS_H", "UDS_K"]
FENIKS_AREA_DEG2 = 0.5801314485383459
# FENIKS_AREA_DEG2 = 0.6081027540413089 #without MegaCam_uS
FENIKS_Z_MIN = 0.2
FENIKS_Z_MAX = 2.5
FENIKS_MAGK_THRESH = 24.0  # tot mag

SDSS_Z_MIN = 0.02
SDSS_Z_MAX = 0.2
SDSS_MAGR_THRESH = 17.5  # model mag

# needs to be total survey area minus the use_phot==0 area
MINERVA_UDS_AREA_DEG2 = 234 / 3600
MINERVA_COSMOS_AREA_DEG2 = 144 / 3600
MINERVA_EGS_AREA_DEG2 = 96 / 3600
MINERVA_AREA_DEG2 = (
    MINERVA_UDS_AREA_DEG2 + MINERVA_COSMOS_AREA_DEG2 + MINERVA_EGS_AREA_DEG2
)

FilterInfo = namedtuple("FilterInfo", ["mag_thresh", "tcurves"])
DatasetLH = namedtuple(
    "DatasetLH",
    [
        "dataset",
        "col_idx",
        "mag_idx",
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
    "ColorColor",
    [
        "parent_cut_idx",
        "col_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "frac_cat",
        "fit",
    ],
)

MagColor = namedtuple(
    "MagColor",
    [
        "parent_cut_idx",
        "mag_idx",
        "col_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "frac_cat",
        "fit",
    ],
)

AppMagFunc = namedtuple(
    "AppMagFunc",
    [
        "parent_cut_idx",
        "mag_idx",
        "sig",
        "bin_lo",
        "bin_hi",
        "N_data",
        "frac_cat",
        "fit",
    ],
)

Lf = namedtuple(
    "Lf",
    ["sig", "bin_lo", "bin_hi", "N_data", "fit"],
)
