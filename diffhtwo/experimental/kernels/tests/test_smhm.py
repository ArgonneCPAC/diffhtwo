import numpy as np
from diffsky.param_utils.diffsky_param_wrapper_merging import DEFAULT_PARAM_COLLECTION

from .. import smhm


def test_median_smhm_and_exsitu_frac(ran_key, fake_subset_ssp_data, feniks_tcurves):
    num_halos = 200
    z_min = 0.2
    z_max = 0.4
    lgmp_min = 10.0
    lgmp_max = 15.0
    d_mh = 0.5

    ssp_data, emline_wave_aa = fake_subset_ssp_data
    (
        logmp_bin_centers,
        logsm_obs_weighted_median,
        logsm_obs_weighted_median_cen_in_situ,
        logsm_obs_weighted_median_cen,
        logsm_obs_weighted_median_sat_in_situ,
        logsm_obs_weighted_median_sat,
        ex_situ_frac_median,
    ) = smhm.median_smhm_and_exsitu_frac(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        feniks_tcurves,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        d_mh=d_mh,
    )
    assert np.isfinite(logmp_bin_centers).all()
    assert np.isfinite(logsm_obs_weighted_median).all()
    assert np.isfinite(logsm_obs_weighted_median_cen_in_situ).all()
    assert np.isfinite(logsm_obs_weighted_median_cen).all()
    assert np.isfinite(logsm_obs_weighted_median_sat_in_situ).all()
    assert np.isfinite(logsm_obs_weighted_median_sat).all()
    assert np.isfinite(ex_situ_frac_median).all()


def test_ex_situ_sm(feniks_lc_phot_data):
    lc_data, phot_data, gal_weight = feniks_lc_phot_data
    logsm_bins = np.arange(9.0, 12.0, 0.5)

    ex_situ_frac_median_sm = smhm.get_ex_situ_frac_median_v_sm(
        logsm_bins,
        phot_data.logsm_obs,
        phot_data.logsm_obs_in_situ,
        gal_weight,
        lc_data.is_central,
    )

    assert np.isfinite(ex_situ_frac_median_sm).all()
