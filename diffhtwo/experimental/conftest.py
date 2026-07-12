from pathlib import Path

import jax.numpy as jnp
import pytest
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from .data_loaders import load_feniks, load_hizels
from .latin_hypercube import lh_utils as lhu
from .lightcone_generators import generate_lc_data
from .utils import load_feniks_tcurve

BASE_PATH = Path(__file__).resolve().parent
FENIKS_DRN = BASE_PATH / "data" / "feniks_test_data"
FENIKS_FILTERS_PATH = BASE_PATH / "data" / "feniks_filters"

HIZELS_DRN = BASE_PATH / "data" / "hizels"


PHOT = "feniks_phot_selected_for_testing.cat"
ZOUT = "feniks_zout_selected_for_testing.ecsv"


@pytest.fixture(scope="session")
def ran_key():
    return jran.key(0)


@pytest.fixture(scope="session")
def fake_subset_ssp_data():
    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
    emline_name = ssp_data.ssp_emline_wave._fields[0]
    emline_wave_aa = ssp_data.ssp_emline_wave[0]
    ssp_data = lemi.get_subset_emline_data(ssp_data, [emline_name])
    return ssp_data, emline_wave_aa


@pytest.fixture(scope="session")
def feniks(ran_key, fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    feniks = load_feniks.get_feniks_data(
        FENIKS_DRN,
        ran_key,
        ssp_data,
        phot=PHOT,
        zout=ZOUT,
        add_random_rows_for_testing=True,
    )
    return feniks


@pytest.fixture(scope="session")
def hizels_fitting_data(ran_key, fake_subset_ssp_data, feniks_tcurves):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    hizels_fitting_data = load_hizels.get_hizels_data(
        HIZELS_DRN, ran_key, ssp_data, feniks_tcurves, emline_wave_aa
    )
    return hizels_fitting_data


@pytest.fixture(scope="session")
def feniks_tcurves():
    tcurves = []
    feniks_filters = ["MegaCam_uS", "HSC_G", "VIDEO_Y", "UDS_K"]
    for feniks_filter in feniks_filters:
        tcurve_filename = FENIKS_FILTERS_PATH / f"{feniks_filter}.txt"
        feniks_filter_wave_aa, feniks_filter_trans = load_feniks_tcurve(tcurve_filename)
        tcurves.append(TransmissionCurve(feniks_filter_wave_aa, feniks_filter_trans))
    return tcurves


@pytest.fixture(scope="session")
def feniks_fitting_data(ran_key, fake_subset_ssp_data):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    feniks_fitting_data = load_feniks.get_feniks_fitting_data(
        FENIKS_DRN,
        ran_key,
        ssp_data,
        phot=PHOT,
        zout=ZOUT,
        add_random_rows_for_testing=False,
        testing=True,
    )
    return feniks_fitting_data


@pytest.fixture(scope="session")
def feniks_single_z_data(ran_key, fake_subset_ssp_data, feniks):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    z_min = 0.2
    z_max = 1.0
    N_centroids = 100

    feniks_meta_data, feniks_fitting_data = lhu.get_single_zbin_lh_lc(
        ran_key,
        feniks,
        z_min,
        z_max,
        ssp_data,
        N_centroids,
    )
    return feniks_meta_data, feniks_fitting_data


@pytest.fixture(scope="session")
def feniks_multi_z_data(ran_key, fake_subset_ssp_data, feniks):
    ssp_data, emline_wave_aa = fake_subset_ssp_data

    feniks_z_min = [0.2, 1]
    feniks_z_max = [1, 2]

    z_mins = feniks_z_min[:2]
    z_maxs = feniks_z_max[:2]

    N_centroids = 100
    num_halos = 100
    feniks_meta_data, feniks_fitting_data = lhu.get_zbins_lh_lc(
        ran_key,
        feniks,
        z_mins,
        z_maxs,
        ssp_data,
        N_centroids,
        num_halos=num_halos,
    )
    return feniks_meta_data, feniks_fitting_data


@pytest.fixture(scope="session")
def feniks_lc_data(ran_key, fake_subset_ssp_data, feniks):
    ssp_data, emline_wave_aa = fake_subset_ssp_data
    tcurves = feniks.filter_info.tcurves

    z_min = 0.2
    z_max = 1.0
    n_z_phot_table = 15
    z_phot_table = 10 ** jnp.linspace(
        jnp.log10(z_min), jnp.log10(z_max), n_z_phot_table
    )

    num_halos = 100
    lgmp_min = 10.0
    lgmp_max = 15.0
    lc_sky_area_degsq = 100

    lc_data = generate_lc_data(
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        lc_sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    return lc_data
