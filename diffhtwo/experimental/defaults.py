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
