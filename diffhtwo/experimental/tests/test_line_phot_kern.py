import numpy as np
from astropy.cosmology import FlatLambdaCDM
from jax.debug import print

from .. import line_phot_kern
from ..data_loaders import retrieve_tcurves
from ..defaults import C_ANGSTROMS, HALPHA_CENTER_AA

HALPHA_LUMINOSITY_CGS = 1e42
REDSHIFT = 0.40  # redshift at which halpha lands at SXDS_z filter
HALPHA_OBS_AA = HALPHA_CENTER_AA * (1 + REDSHIFT)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0474, Tcmb0=2.7255)
D_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm


SXDS_z_tcurve = retrieve_tcurves.SXDS_z
SXDS_z_tcurve_wave_aa = SXDS_z_tcurve[:, 0]
SXDS_z_tcurve_trans = SXDS_z_tcurve[:, 1]


def test_line_phot_kern(
    HALPHA_LUMINOSITY_CGS=HALPHA_LUMINOSITY_CGS,
    REDSHIFT=REDSHIFT,
    HALPHA_OBS_AA=HALPHA_OBS_AA,
    D_L=D_L,
):
    halpha_flux_app_cgs = line_phot_kern._flux_app_from_luminosity(
        HALPHA_LUMINOSITY_CGS, REDSHIFT, D_L
    )
    assert np.isfinite(halpha_flux_app_cgs)
    assert halpha_flux_app_cgs < HALPHA_LUMINOSITY_CGS

    SXDS_z_equivalent_width_aa = line_phot_kern._tcurve_equivalent_width(
        SXDS_z_tcurve_wave_aa, SXDS_z_tcurve_trans
    )
    assert np.isfinite(SXDS_z_equivalent_width_aa)

    halpha_flux_density_filter_aa = line_phot_kern.flux_density_filter_aa(
        HALPHA_OBS_AA,
        halpha_flux_app_cgs,
        SXDS_z_tcurve_wave_aa,
        SXDS_z_tcurve_trans,
        SXDS_z_equivalent_width_aa,
    )
    assert np.isfinite(halpha_flux_density_filter_aa)

    halpha_flux_density_filter_hz = line_phot_kern._flux_density_aa_to_hz(
        halpha_flux_density_filter_aa, HALPHA_OBS_AA
    )
    assert np.isfinite(halpha_flux_density_filter_hz)

    mag_ab = line_phot_kern._flux_density_hz_to_mag_ab(halpha_flux_density_filter_hz)
    assert mag_ab > 15

    # calculate mag_ab with minimal use of line_phot_kern.py functions
    d_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm
    T = np.interp(HALPHA_OBS_AA, SXDS_z_tcurve_wave_aa, SXDS_z_tcurve_trans)
    assert (T >= 0) & (T <= 1)
    F_halpha = HALPHA_LUMINOSITY_CGS / (4 * np.pi * (d_L**2))
    numerator = T * F_halpha * (HALPHA_OBS_AA**2)
    denominator = SXDS_z_equivalent_width_aa * C_ANGSTROMS
    mag_ab_check = -2.5 * np.log10(numerator / denominator) - 48.6

    print("mag_ab={}", mag_ab)
    print("mag_ab_check={}", mag_ab_check)
    assert np.close(mag_ab, mag_ab_check)
