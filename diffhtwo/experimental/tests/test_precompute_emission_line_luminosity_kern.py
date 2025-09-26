import jax.numpy as jnp
import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data

from ..data_loaders import retrieve_fake_fsps_halpha
from ..precompute_emission_line_luminosity_kern import (
    get_emission_line_luminosity,
)

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_halpha_line_luminosity = retrieve_fake_fsps_halpha.load_fake_ssp_halpha()


def test_precompute_fsps_halpha_luminosity(
    CONTINUUM_FIT_LO_LO=6480,
    CONTINUUM_FIT_LO_HI=6530,
    CONTINUUM_FIT_HI_LO=6600,
    CONTINUUM_FIT_HI_HI=6650,
    HALPHA_LINE_CENTER=6564.5131,  # halpha center wavelength in Angstroms for FSPS,
    HALPHA_LINE_DELTA=5,  # wavelength width to capture halpha line luminosity,
):
    # needed for some like bc03_pdva_stelib_chabrier.h5
    ssp_wave = np.float64(ssp_data.ssp_wave)

    n_met = ssp_data.ssp_lgmet.shape[0]
    n_age = ssp_data.ssp_lg_age_gyr.shape[0]

    line_lo = HALPHA_LINE_CENTER - HALPHA_LINE_DELTA
    line_hi = HALPHA_LINE_CENTER + HALPHA_LINE_DELTA
    line_sel = (ssp_wave >= line_lo) & (ssp_wave <= line_hi)

    # Test edge case when sed dips below continuum within line window
    ssp_flux = -13.5 * np.ones(ssp_wave.shape)
    ssp_flux[line_sel] = -14.0

    losses, theta, sed_dict = get_emission_line_luminosity(
        ssp_wave,
        ssp_flux,
        CONTINUUM_FIT_LO_LO,
        CONTINUUM_FIT_LO_HI,
        CONTINUUM_FIT_HI_LO,
        CONTINUUM_FIT_HI_HI,
        HALPHA_LINE_CENTER,
        HALPHA_LINE_DELTA,
    )
    assert sed_dict["line_integrated_luminosity"] == 0.0

    for i in range(5):
        met = np.random.randint(0, n_met)
        age = np.random.randint(0, n_age)
        ssp_flux = jnp.log10(ssp_data.ssp_flux[met][age])
        losses, theta, sed_dict = get_emission_line_luminosity(
            ssp_wave,
            ssp_flux,
            CONTINUUM_FIT_LO_LO,
            CONTINUUM_FIT_LO_HI,
            CONTINUUM_FIT_HI_LO,
            CONTINUUM_FIT_HI_HI,
            HALPHA_LINE_CENTER,
            HALPHA_LINE_DELTA,
        )

        line_luminosity = ssp_data.ssp_flux[met][age][line_sel]
        ssp_wave_max = ssp_wave[line_sel][line_luminosity.argmax()]

        if sed_dict["line_integrated_luminosity"] != 0.0:
            assert np.isclose(ssp_wave_max, HALPHA_LINE_CENTER)

        assert np.isfinite(sed_dict["line_integrated_luminosity"])
        assert sed_dict["line_integrated_luminosity"] >= 0.0
