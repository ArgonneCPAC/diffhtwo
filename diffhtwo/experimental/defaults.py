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
FENIKS_Z_MAX = 4.0
FENIKS_MAGK_THRESH = 24.3  # col mag

SDSS_AREA_DEG2 = 7199
SDSS_Z_MIN = 0.02
SDSS_Z_MAX = 0.2
SDSS_MAGR_THRESH = 17.6  # model mag


DATASET = namedtuple(
    "DATASET",
    [
        "dataset",
        "mags",
        "tcurves",
        "mag_columns",
        "mag_thresh_column",
        "mag_thresh",
        "frac_cat",
        "lh_centroids",
        "d_centroids",
        "N_data",
        "data_sky_area_degsq",
        "lh_dmag",
        "lh_dz",
    ],
)
