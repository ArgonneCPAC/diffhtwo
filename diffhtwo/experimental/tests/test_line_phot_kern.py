import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental.scatter import DEFAULT_SCATTER_PARAMS
from diffsky.param_utils import spspop_param_utils as spspu
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.metallicity import umzr
from jax import random as jran

from .. import line_phot_kern
from ..data_loaders import retrieve_fake_fsps_halpha, retrieve_tcurves
from ..defaults import C_ANGSTROMS, HALPHA_CENTER_AA
from ..diffstarpop_halpha import (
    diffstarpop_halpha_kern,
    diffstarpop_halpha_lf_weighted_lc_weighted,
)

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_halpha_line_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()

HALPHA_LUMINOSITY_CGS = 1e42

# Redshift from which HALPHA_CENTER_AA emitted
# lands at HSC_NB921 filter peak transmission at 9213.2 Å
REDSHIFT = 0.4035

HALPHA_OBS_AA = HALPHA_CENTER_AA * (1 + REDSHIFT)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0474, Tcmb0=2.7255)
D_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm

DISTANCE_MPC = 1 * u.Mpc
MPC_TO_CM = DISTANCE_MPC.to(u.cm).value


SXDS_z_tcurve = retrieve_tcurves.SXDS_z
HSC_NB921_tcurve = retrieve_tcurves.HSC_NB921


def test_line_phot_kern(BB_tcurve=SXDS_z_tcurve, NB_tcurve=HSC_NB921_tcurve):
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

    # mag should be brighter in narrow-band vs. broad-band because
    # emission line signal gets diluted in broad-band due to much larger
    # equivalent width
    assert NB_mag_ab < BB_mag_ab

    # calculate mag_ab with minimal use of line_phot_kern.py functions
    d_L = COSMO.luminosity_distance(REDSHIFT).to("cm").value  # Mpc to cm
    T = np.interp(HALPHA_OBS_AA, BB_tcurve_wave_aa, BB_tcurve_trans)
    assert (T >= 0) & (T <= 1)
    F_halpha = HALPHA_LUMINOSITY_CGS / (4 * np.pi * (d_L**2))
    numerator = T * F_halpha * (HALPHA_OBS_AA**2)
    denominator = BB_equivalent_width_aa * C_ANGSTROMS
    BB_mag_ab_check = -2.5 * np.log10(numerator / denominator) - 48.6

    assert np.isclose(BB_mag_ab, BB_mag_ab_check)


def test_line_mag_vmap():
    ran_key = jran.key(0)
    ran_key, lc_key = jran.split(ran_key, 2)

    lgmp_min = 12.0
    z_min, z_max = 0.2, 0.5
    sky_area_degsq = 1

    """weighted mc lightcone"""
    num_halos = 500
    lgmp_max = 15.0
    args = (lc_key, num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    lc_halopop = mclh.mc_weighted_halo_lightcone(*args)

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    z_obs = jnp.array(lc_halopop["z_obs"])
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    # logmp0 = lc_halopop["logmp0"]
    logmp0 = lc_halopop["logmp0"]
    nhalos = lc_halopop["nhalos"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    scatter_params = DEFAULT_SCATTER_PARAMS

    ran_key, dpop_halpha_true_key = jran.split(ran_key, 2)
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        dpop_halpha_true_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        ssp_halpha_line_luminosity,
        z_phot_table,
        mzr_params,
        spspop_params,
        scatter_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    halpha_L_true = diffstarpop_halpha_kern(*args)

    (
        lgL_bin_edges,
        halpha_lf_weighted_q_true,
        halpha_lf_weighted_smooth_ms_true,
        halpha_lf_weighted_bursty_ms_true,
    ) = diffstarpop_halpha_lf_weighted_lc_weighted(halpha_L_true, nhalos)

    halpha_obs_aa = HALPHA_CENTER_AA * (1 + z_obs)
    d_L_Mpc = COSMO.luminosity_distance(z_obs)
    d_L_cm = d_L_Mpc * MPC_TO_CM

    SXDS_z_wave_aa = SXDS_z_tcurve[:, 0]
    SXDS_z_trans = SXDS_z_tcurve[:, 1]

    SXDS_z_mag_ab = line_phot_kern.get_band_mag_ab_from_luminosity(
        halpha_obs_aa, halpha_L_true, z_obs, d_L_cm, SXDS_z_wave_aa, SXDS_z_trans
    )

    assert np.isfinite(SXDS_z_mag_ab.band_mag_ab_q).all()
    assert np.isfinite(SXDS_z_mag_ab.band_mag_ab_smooth_ms).all()
    assert np.isfinite(SXDS_z_mag_ab.band_mag_ab_bursty_ms).all()
