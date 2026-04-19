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

# FENIKS_FRAC_CAT is the fraction of galaxies remaining after the initial
# !=-99 cut in all used bands, and further removal of objects that had NaNs
# after taking log10 of flux for mag calculation
FENIKS_FRAC_CAT = 0.6379076134281674

FENIKS_AREA_DEG2 = 2828.247933129912 / 3600
FENIKS_Z_MIN = 0.2
FENIKS_Z_MAX = 4.0
FENIKS_MAGK_THRESH = 24.3

SDSS_FRAC_CAT = 0.8191344722947992
SDSS_AREA_DEG2 = 8000
SDSS_Z_MIN = 0.02
SDSS_Z_MAX = 0.2
SDSS_MAGR_THRESH = 17.7


DATASET = namedtuple(
    "DATASET",
    [
        "dataset",
        "tcurves",
        "mag_columns",
        "mag_thresh_column",
        "mag_thresh",
        "frac_cat",
        "lh_centroids",
        "d_centroids",
        "lg_n_data_err_lh",
        "lc_data",
    ],
)
