import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .. import line_phot_kern
from ..data_loaders import retrieve_tcurves
from ..defaults import C_ANGSTROMS, HALPHA_CENTER_AA

HALPHA_LUMINOSITY_CGS = 1e42
REDSHIFT = 0.40  # redshift at which halpha lands at SXDS_z filter
HALPHA_OBS_AA = HALPHA_CENTER_AA * (1 + REDSHIFT)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0474, Tcmb0=2.7255)
D_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm


SXDS_z_tcurve = retrieve_tcurves.SXDS_z
HSC_NB921_tcurve = retrieve_tcurves.HSC_NB921


def test_line_phot_kern(
	BB_tcurve=SXDS_z_tcurve,
	NB_tcurve=HSC_NB921_tcurve
):
	BB_tcurve_wave_aa = BB_tcurve[:, 0]
	BB_tcurve_trans = BB_tcurve[:, 1]

	NB_tcurve_wave_aa = NB_tcurve[:, 0]
	NB_tcurve_trans = NB_tcurve[:, 1]

	
    halpha_flux_app_cgs = line_phot_kern._flux_app_from_luminosity(
        HALPHA_LUMINOSITY_CGS, REDSHIFT, D_L
    )
    assert np.isfinite(halpha_flux_app_cgs)
    assert halpha_flux_app_cgs < HALPHA_LUMINOSITY_CGS


    # BB equivalent_width
    BB_equivalent_width_aa = line_phot_kern._tcurve_equivalent_width(
        BB_tcurve_wave_aa, BB_tcurve_trans
    )
    assert np.isfinite(BB_equivalent_width_aa)

    # NB equivalent_width
    NB_equivalent_width_aa = line_phot_kern._tcurve_equivalent_width(
        NB_tcurve_wave_aa, NB_tcurve_trans
    )
    assert np.isfinite(NB_equivalent_width_aa)


    # BB flux_density_aa
    BB_flux_density_filter_aa = line_phot_kern.flux_density_filter_aa(
        HALPHA_OBS_AA,
        halpha_flux_app_cgs,
        BB_tcurve_wave_aa,
        BB_tcurve_trans,
        BB_equivalent_width_aa,
    )
    assert np.isfinite(BB_flux_density_filter_aa)

    # NB flux_density_aa
    NB_flux_density_filter_aa = line_phot_kern.flux_density_filter_aa(
        HALPHA_OBS_AA,
        halpha_flux_app_cgs,
        NB_tcurve_wave_aa,
        NB_tcurve_trans,
        NB_equivalent_width_aa,
    )
    assert np.isfinite(NB_flux_density_filter_aa)


    # BB flux_density_hz
    BB_flux_density_filter_hz = line_phot_kern._flux_density_aa_to_hz(
        BB_flux_density_filter_aa, HALPHA_OBS_AA
    )
    assert np.isfinite(BB_flux_density_filter_hz)

    # NB flux_density_hz
    NB_flux_density_filter_hz = line_phot_kern._flux_density_aa_to_hz(
        NB_flux_density_filter_aa, HALPHA_OBS_AA
    )
    assert np.isfinite(NB_flux_density_filter_hz)


    # BB mag_ab
    BB_mag_ab = line_phot_kern._flux_density_hz_to_mag_ab(BB_flux_density_filter_hz)
    assert BB_mag_ab > 15

    # NB mag_ab
    NB_mag_ab = line_phot_kern._flux_density_hz_to_mag_ab(NB_flux_density_filter_hz)
    assert NB_mag_ab > 15

    # calculate mag_ab with minimal use of line_phot_kern.py functions
    d_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm
    T = np.interp(HALPHA_OBS_AA, BB_tcurve_wave_aa, BB_tcurve_trans)
    assert (T >= 0) & (T <= 1)
    F_halpha = HALPHA_LUMINOSITY_CGS / (4 * np.pi * (d_L**2))
    numerator = T * F_halpha * (HALPHA_OBS_AA**2)
    denominator = BB_equivalent_width_aa * C_ANGSTROMS
    BB_mag_ab_check = -2.5 * np.log10(numerator / denominator) - 48.6

    assert np.isclose(BB_mag_ab, BB_mag_ab_check)
