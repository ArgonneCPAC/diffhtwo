from pathlib import Path

import numpy as np
import pytest
from diffsky.param_utils.diffsky_param_wrapper_merging import (
    DEFAULT_PARAM_COLLECTION,
    check_param_collection_is_ok,
)
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...utils import load_feniks_tcurve
from ..lc_phot_kern import mc_phot_kern_merging_wrapper, multiband_lc_phot_kern

BASE_PATH = Path(__file__).resolve().parent.parent.parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


@pytest.fixture(scope="module")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


def test(fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    ran_key = jran.key(0)
    z_min = 0.2
    z_max = 0.5
    num_halos = 100

    tcurves = []
    feniks_filters = ["MegaCam_uS", "HSC_G", "VIDEO_Y", "UDS_K"]
    for feniks_filter in feniks_filters:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))

    assert check_param_collection_is_ok(DEFAULT_PARAM_COLLECTION)
    lc_data, phot_kern_results, gal_weight = multiband_lc_phot_kern(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        z_min,
        z_max,
        num_halos,
        ssp_data,
        tcurves,
    )
    assert np.isfinite(lc_data.cen_weight).all()
    assert np.isfinite(lc_data.sat_weight).all()
    assert np.isfinite(gal_weight).all()
    assert np.isfinite(phot_kern_results.obs_mags).all()

    phot_kern_results2 = mc_phot_kern_merging_wrapper(
        ran_key,
        DEFAULT_PARAM_COLLECTION,
        lc_data,
    )
    assert np.isfinite(phot_kern_results2.obs_mags).all()
