from pathlib import Path

import numpy as np
from dsps.data_loaders import load_transmission_curve

from ...data_loaders.load_minerva import MINERVA_FILTERS_PATH, PhotFilters
from ...lightcone_generators import generate_lc_data

BASE_PATH = Path(__file__).resolve().parent.parent.parent
IGM_DRN = BASE_PATH / "data" / "igm"
IGM_BN = "igm_attenuation_minerva.h5"


def test_igm_attn_ssp_mag_table(ran_key, fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    n_host_halos = 100
    lgmp_min = 11
    lgmp_max = 15
    sky_area_degsq = 10
    z_min = 4
    z_max = 4.2
    z_phot_table = np.linspace(z_min, z_max, 15)

    tcurves = []
    for minerva_filter in PhotFilters._fields:
        tcurve = load_transmission_curve(
            bn_pat=minerva_filter + "*", drn=MINERVA_FILTERS_PATH
        )
        tcurves.append(tcurve)

    lc_data_w_igm = generate_lc_data(
        ran_key,
        n_host_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
        apply_igm=True,
        igm_drn=IGM_DRN,
        igm_bn=IGM_BN,
        igm_filters_namedtuple=PhotFilters,
        igm_filter_prefix="minerva_",
        logmp_cutoff=lgmp_min,
    )
    precomputed_ssp_mag_table_w_igm = lc_data_w_igm.precomputed_ssp_mag_table

    lc_data_no_igm = generate_lc_data(
        ran_key,
        n_host_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
        apply_igm=False,
    )
    precomputed_ssp_mag_table_no_igm = lc_data_no_igm.precomputed_ssp_mag_table

    assert np.isfinite(precomputed_ssp_mag_table_w_igm).all()
    assert np.isfinite(precomputed_ssp_mag_table_no_igm).all()
    assert np.all(precomputed_ssp_mag_table_w_igm >= precomputed_ssp_mag_table_no_igm)
