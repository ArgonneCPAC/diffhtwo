import numpy as np

from ..fq import get_fq_hm, get_fq_sm
from ..sfr_tau import get_logsfr_100Myr


def test_fq_sm_hm(feniks_lc_phot_data, fake_subset_ssp_data):
    lc_data, phot_data, gal_weight = feniks_lc_phot_data
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    logsfr_100Myr = get_logsfr_100Myr(phot_data, lc_data, ssp_data)
    d_sm = 0.5
    d_hm = 0.5

    f_q_sm, logsm_bin_centers = get_fq_sm(
        phot_data.logsm_obs,
        logsfr_100Myr,
        lc_data,
        phot_data,
        gal_weight,
        type="all",
        quench_thresh=-11,
        d_sm=d_sm,
    )
    assert np.isfinite(f_q_sm).all()

    f_q_hm, logmp_bin_centers = get_fq_hm(
        phot_data.logsm_obs,
        logsfr_100Myr,
        lc_data,
        phot_data,
        gal_weight,
        type="all",
        quench_thresh=-11,
        d_hm=d_hm,
    )
    assert np.isfinite(f_q_hm).all()
