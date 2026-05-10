from pathlib import Path

import pytest
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve

from .utils import load_feniks_tcurve

BASE_PATH = Path(__file__).resolve().parent
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"


@pytest.fixture(scope="session")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


@pytest.fixture(scope="session")
def feniks_tcurves():
    tcurves = []
    feniks_filters = ["MegaCam_uS", "HSC_G", "VIDEO_Y", "UDS_K"]
    for feniks_filter in feniks_filters:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))
    return tcurves
